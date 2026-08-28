"""Extract hidden-state embeddings from an arbitrary causal LM for arbitrary prompts.

Prompt sources:
  --prompts FILE   plain .txt (one per line) or .jsonl ({"id":..,"text":..})
  --template FILE  a prompt template with {placeholders}; values are pulled per row
                   from --stimuli (an .xlsx), tracking each bracket's char span.

Counterbalanced two-sentence prompts:
  --counterbalance A B  Run the extraction twice in one invocation (single model load).
                        The first pass uses the template as-is and writes to
                        outputs/<model>/<set_name>_fwd/. The second pass swaps the
                        {A} and {B} placeholders in the prompt text and writes to
                        outputs/<model>/<set_name>_rev/. Bracket names follow the
                        original placeholder names; the swap only changes their
                        position in the prompt. Use for counterbalancing semantic
                        candidate sentences (e.g. --counterbalance I PU).

Pooling (per layer):
  --pooling whole      mean over all prompt tokens
            last       the last (answer-position) token, padding-safe
            all        every token (ragged -> flat [sum_tokens, d] + token index)
  --brackets all|F...  per-bracket span pooling (mean over each field's tokens);
                       requires --template and a fast tokenizer.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import hashlib
import json
import tempfile
import time

import numpy as np

from core import io as embeddings_io
import config
import registry

POOLINGS = ("whole", "last", "all")
# Bracket pooling modes:
#   mean        - uniform mean over the span tokens (default).
#   last        - the final token of the span.
#   bertscore   - IDF-weighted mean over the span tokens. IDF is computed per bracket
#                 field as document-frequency over the bracket spans of THIS run:
#                 idf(t) = ln((N+1)/(df(t)+1))  (BERTScore/Zhang-2019 def). Down-weights tokens
#                 that appear in many stimuli (function words, repeated frame tokens),
#                 up-weights distinctive content tokens. Matches the IDF weighting
#                 introduced by BERTScore (Zhang et al. 2019), applied as a pooling.
BRACKET_POOLINGS = ("mean", "last", "bertscore")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Alias in models.csv, or a raw HF repo id.")
    p.add_argument("--models-config", default=None, help="Unused; kept for backward compatibility.")
    p.add_argument("--prompts", default=None, help=".txt (one per line) or .jsonl ({id,text}).")
    p.add_argument("--prompt", action="append", default=[], help="Inline prompt; repeatable.")
    p.add_argument("--template", default=None, help="Template with {placeholders} (enables --brackets).")
    p.add_argument("--stimuli", default=str(config.STIMULI / "items_648.xlsx"),
                   help="Workbook supplying values for the template placeholders. "
                        "Default columns: QUD, U, PU, I, CT, CT2, CT3, CT4, NR (plus QUD_polar). "
                        "Use any column name as a {placeholder} in the prompt template.")
    p.add_argument("--field-map", default=None, help="JSON mapping placeholder->column, e.g. '{\"answer\":\"S2\"}'.")
    p.add_argument("--sentences", nargs="+", default=None,
                   help="Carrier mode: which column(s) fill the {sentence} slot, one prompt each: "
                        "'all' or names like PU I CT (case-insensitive). Requires --template with the slot.")
    p.add_argument("--sentence-slot", default="sentence", help="Placeholder name of the carrier slot.")
    p.add_argument("--counterbalance", nargs=2, metavar=("A", "B"), default=None,
                   help="Run two passes from one model load, swapping placeholders {A} and {B} in "
                        "the second pass. Writes <set_name>_fwd and <set_name>_rev.")
    p.add_argument("--layers", nargs="+", default=["-1"], help="hidden_states indices; negatives / 'all' ok.")
    p.add_argument("--pooling", nargs="+", default=["whole"], choices=POOLINGS)
    p.add_argument("--brackets", nargs="+", default=None,
                   help="'all' or specific placeholder names to pool per-bracket (needs --template).")
    p.add_argument("--bracket-pooling", nargs="+", default=["mean"], choices=BRACKET_POOLINGS,
                   help="How to pool each bracket span. 'mean' = mean over the span; "
                        "'last' = the final token of the span. Pass both to save both.")
    p.add_argument("--bertscore-pairs", nargs="+", default=None,
                   help="Compute cross-layer BERTScore F1 between bracket spans, per stimulus. "
                        "'auto' = all C(K,2) pairs of --brackets; or list pairs A:B C:D. "
                        "Saved as bertscore_pairs.safetensors with keys '<A>__<B>__F1' of shape "
                        "[N, L_A, L_B]. Default: disabled.")
    p.add_argument("--bertscore-no-idf", action="store_true",
                   help="Disable IDF weighting in BERTScore (default: IDF on, per BERTScore paper).")
    p.add_argument("--bertscore-save-pr", action="store_true",
                   help="Also save the asymmetric Precision and Recall components (~3x storage).")
    p.add_argument("--save-span-tokens", action="store_true",
                   help="Case A capture for curvature: save the PER-TOKEN hidden states of every "
                        "bracket-span token (all layers), STREAMED per batch to span_tokens.bin "
                        "(raw [rows, n_layers, d], memmap-loadable; shape in span_tokens_meta.json) "
                        "+ span_tokens_index.parquet (row, prompt_id, span, pos_in_span, token_pos, "
                        "token_id, token_str). Peak RAM ~ one batch. Requires --brackets + a fast "
                        "tokenizer. LARGE on disk: ~60-75 tokens/stimulus x all layers.")
    p.add_argument("--last-prompt-tokens", type=int, default=0,
                   help="With --save-span-tokens, also keep the last K attended prompt tokens per "
                        "stimulus (answer region), tagged span='lastK'.")
    p.add_argument("--span-fp16", action="store_true",
                   help="Store span_tokens in fp16 (halves RAM + disk), clamped to +/-65504 to "
                        "avoid overflow. Safe for Qwen/Llama/OLMo/Mistral; keep fp32 for Gemma "
                        "(massive deep-layer activations).")
    p.add_argument("--span-entropy", action="store_true",
                   help="With --save-span-tokens, also store the full-vocabulary next-token "
                        "Shannon entropy (nats) at each captured token position, as column "
                        "next_tok_entropy in span_tokens_index.parquet. This is the paper's R2 "
                        "target H(w_n): entropy of softmax(logits at that position), a per-position "
                        "output-distribution scalar (NOT a per-layer quantity). Forces the model to "
                        "emit all-position logits (logits_to_keep=0), so peak GPU memory rises by "
                        "[batch, seq, vocab]; entropy is reduced at the captured positions only.")
    p.add_argument("--raw", action="store_true", help="Skip the chat template (tokenize the prompt as-is).")
    p.add_argument("--think", action="store_true",
                   help="Keep model 'thinking' ON in the chat template. Default OFF: this is a "
                        "single-token forced-choice extraction, so thinking is disabled "
                        "(enable_thinking=False where the template supports it) to keep the answer "
                        "position on the choice token and match the behavioural reference condition.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--load-4bit", action="store_true",
                   help="Load weights in 4-bit (nf4, compute dtype = --dtype); forces device_map=auto. "
                        "WARNING: distorts behaviour on small models (Qwen2.5-0.5B collapses) and the "
                        "hidden states — only for memory-bound sanity checks, never for trusted results.")
    p.add_argument("--max-mem-per-gpu", type=str, default=None,
                   help="device_map=auto only: cap weights per visible GPU (e.g. '5GiB'), reserving "
                        "activation headroom so a bf16 model spreads across several small cards without OOM.")
    p.add_argument("--device", default="auto", help="'auto' (device_map) or e.g. 'cuda:0' / 'cpu'.")
    p.add_argument("--attn-impl", default=None, choices=["eager", "sdpa", "flash_attention_2"],
                   help="attn_implementation override. 'eager' upcasts softmax to fp32 and avoids the "
                        "SDPA bf16 NaN seen with device_map=auto sharding. Default: HF auto (sdpa).")
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-answer", action="store_true")
    p.add_argument("--answer-tokens", nargs="+", default=None,
                   help="Restrict the recorded answer to these option strings (e.g. '0 1'): take the "
                        "argmax over ONLY their token logits instead of the full vocab. Use for models "
                        "that emit non-answer tokens (e.g. gemma-4-12B's <audio|> special token) so a "
                        "spurious top token can't win. answer_prob stays the full-vocab prob of the pick.")
    p.add_argument("--output-dir", default=str(config.REPR_OUTPUTS),
                   help="Root for per-token activation outputs (default: config.REPR_OUTPUTS).")
    p.add_argument("--set-name", default=None)
    return p.parse_args()


def resolve_model(spec: str, config_path: str | None = None) -> tuple[str, str, bool]:
    """Resolve a model spec to (alias, hf_id, use_chat).

    ``spec`` may be an alias listed in models.csv or a raw Hugging Face repo id.
    """
    if spec in registry.HF_ID:
        return spec, registry.HF_ID[spec], True
    return spec.split("/")[-1], spec, True


def load_plain_prompts(args) -> list[dict]:
    items: list[dict] = []
    if args.prompts:
        path = Path(args.prompts)
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            for i, line in enumerate(text.splitlines()):
                if line.strip():
                    obj = json.loads(line)
                    items.append({"id": str(obj.get("id", i)), "text": obj["text"], "field_spans": {}})
        else:
            for i, line in enumerate(text.splitlines()):
                if line.strip():
                    items.append({"id": str(i), "text": line, "field_spans": {}})
    for j, t in enumerate(args.prompt):
        items.append({"id": f"inline-{j}", "text": t, "field_spans": {}})
    if not items:
        raise SystemExit("No prompts. Use --prompts/--prompt, or --template with --stimuli.")
    return items


def resolve_layers(spec: list[str], n_states: int) -> list[int]:
    if len(spec) == 1 and spec[0].lower() == "all":
        return list(range(n_states))
    out = set()
    for tok in spec:
        i = int(tok)
        i = i + n_states if i < 0 else i
        if not 0 <= i < n_states:
            raise SystemExit(f"Layer {tok} out of range; model has {n_states} hidden states (0..{n_states-1}).")
        out.add(i)
    return sorted(out)


def last_token_indices(attention_mask):
    """Last attended token per row, robust to left/right padding."""
    seq_len = attention_mask.shape[1]
    return seq_len - 1 - attention_mask.flip(dims=[1]).argmax(dim=1)


def swap_placeholders(text: str, a: str, b: str) -> str:
    """Swap two placeholder names in a template, e.g. {I} <-> {PU}. Two-step swap via
    a sentinel so the second .replace doesn't undo the first."""
    sentinel = "\x00SWAP\x00"
    text = text.replace("{" + a + "}", "{" + sentinel + "}")
    text = text.replace("{" + b + "}", "{" + a + "}")
    text = text.replace("{" + sentinel + "}", "{" + b + "}")
    return text


def _build_items_for_pass(args, template_path: str | None):
    """Build items + return (items, template_fields, carrier_mode). Stays self-contained
    so each counterbalance pass gets its own item set built from its own template text."""
    template_fields: list[str] = []
    carrier_mode = bool(args.sentences)
    if carrier_mode and not template_path:
        raise SystemExit("--sentences requires --template (a carrier with a {sentence} slot).")
    if template_path:
        from core import stimuli
        field_map = json.loads(args.field_map) if args.field_map else None
        if carrier_mode:
            items, template_fields = stimuli.build_carrier_items(
                template_path, args.stimuli, args.sentences, args.sentence_slot, field_map, args.limit)
        else:
            items, template_fields = stimuli.build_items(template_path, args.stimuli, field_map, args.limit)
    else:
        items = load_plain_prompts(args)
        if args.limit is not None:
            items = items[: args.limit]
    return items, template_fields, carrier_mode


def run_pass(
    args,
    *,
    template_path: str | None,
    set_name: str | None,
    pass_label: str,
    # Pre-loaded resources, shared across counterbalance passes:
    torch, transformers, pd, model, tok, device,
    alias: str, hf_id: str, use_chat: bool, n_states: int, layers: list[int],
    poolings: list[str], bracket_modes: list[str],
) -> None:
    """One full extraction pass: build items from template_path, run model forward,
    save to outputs/<alias>/<set_name>/. The model is NOT loaded here; caller passes it in."""
    import math

    if pass_label:
        print(f"\n=== Pass {pass_label} → set_name={set_name} ===", flush=True)

    items, template_fields, carrier_mode = _build_items_for_pass(args, template_path)

    bracket_fields: list[str] = []
    if args.brackets:
        if not template_path:
            raise SystemExit("--brackets requires --template.")
        bracket_fields = template_fields if "all" in args.brackets else args.brackets
        unknown = [f for f in bracket_fields if f not in template_fields]
        if unknown:
            raise SystemExit(f"--brackets {unknown} not in template fields {template_fields}.")
    elif carrier_mode:
        bracket_fields = template_fields
    bracket_on = bool(bracket_fields)
    if args.save_span_tokens and not bracket_on:
        raise SystemExit("--save-span-tokens requires --brackets (needs the span token masks).")
    if args.span_entropy and not args.save_span_tokens:
        raise SystemExit("--span-entropy requires --save-span-tokens (entropy is stored per span token).")
    if bracket_on and not tok.is_fast:
        raise SystemExit("--brackets needs a fast tokenizer (offset mapping); this model has a slow one.")

    # Thinking is OFF by default for this forced-choice extraction (see --think). Pass
    # enable_thinking=False where the chat template accepts it; fall back cleanly otherwise
    # (non-thinking templates ignore/reject the kwarg). Decided once, reused per item.
    chat_kwargs = {"add_generation_prompt": True, "tokenize": False}
    thinking = "n/a (no chat template)"
    if use_chat:
        if args.think:
            thinking = "enabled (--think)"
        else:
            try:
                tok.apply_chat_template([{"role": "user", "content": "x"}],
                                        enable_thinking=False, add_generation_prompt=True, tokenize=False)
                chat_kwargs["enable_thinking"] = False
                thinking = "disabled (enable_thinking=False)"
            except (TypeError, ValueError):
                thinking = "off by default; template has no enable_thinking knob"

    print(f"{len(items)} prompts | layers {layers} | pooling {poolings} | "
          f"brackets {bracket_fields or '-'} | chat_template={use_chat} | thinking={thinking}", flush=True)

    def render(item):
        raw, spans = item["text"], item.get("field_spans", {})
        if use_chat:
            s = tok.apply_chat_template([{"role": "user", "content": raw}], **chat_kwargs)
            base = s.find(raw)
            if base >= 0:
                spans = {f: (a + base, b + base) for f, (a, b) in spans.items()}
            return s, spans
        return raw, spans

    finals = [render(it) for it in items]

    pooled = {p: {l: [] for l in layers} for p in poolings if p in ("whole", "last")}
    tokens_all = {l: [] for l in layers} if "all" in poolings else {}
    save_span = args.save_span_tokens
    span_index_rows = []
    span_total = 0            # rows written so far (global row id)
    span_dim = None           # feature dim d
    span_fh = None            # streaming file handle for span_tokens.bin
    span_run_dir = None
    if save_span:
        _rs = set_name or (Path(template_path).stem if template_path
                           else Path(args.prompts).stem if args.prompts
                           else time.strftime("run_%Y%m%d_%H%M%S"))
        span_run_dir = Path(args.output_dir) / alias / _rs
        span_run_dir.mkdir(parents=True, exist_ok=True)
        span_fh = open(span_run_dir / "span_tokens.bin", "wb")   # stream: [rows, n_layers, d]
    bracket_acc = {m: {f: {l: [] for l in layers} for f in bracket_fields} for m in bracket_modes}
    bracket_cnt = {f: [] for f in bracket_fields}
    tok_index_rows, meta_rows = [], []

    # ---- BERTScore pair selection ----
    # Parse --bertscore-pairs into a list of (A, B) field tuples; default to nothing.
    bs_pairs: list[tuple[str, str]] = []
    if args.bertscore_pairs:
        if not bracket_on:
            raise SystemExit("--bertscore-pairs requires --brackets.")
        if args.bertscore_pairs == ["auto"]:
            bs_pairs = [(a, b) for i, a in enumerate(bracket_fields)
                        for b in bracket_fields[i + 1:]]
        else:
            for spec in args.bertscore_pairs:
                if ":" not in spec:
                    raise SystemExit(f"--bertscore-pairs entry {spec!r} must be 'auto' or 'A:B'.")
                a, b = spec.split(":", 1)
                if a not in bracket_fields or b not in bracket_fields:
                    raise SystemExit(f"--bertscore-pairs {spec}: both {a} and {b} must appear "
                                     f"in --brackets ({bracket_fields}).")
                bs_pairs.append((a, b))
        print(f"BERTScore pairs: {bs_pairs}  (IDF={'off' if args.bertscore_no_idf else 'on'}, "
              f"save_PR={args.bertscore_save_pr})", flush=True)

    bs_acc_F1 = {pair: [] for pair in bs_pairs}
    bs_acc_P = {pair: [] for pair in bs_pairs} if args.bertscore_save_pr else None
    bs_acc_R = {pair: [] for pair in bs_pairs} if args.bertscore_save_pr else None

    # ---- IDF computation ----
    # Used by both 'bertscore' bracket pooling AND BERTScore-pairs (unless --bertscore-no-idf).
    # Computed once via a tokenizer pass over all prompts; per-field.
    idf_lookup: dict[str, "torch.Tensor"] = {}
    need_idf = ("bertscore" in bracket_modes) or (bs_pairs and not args.bertscore_no_idf)
    if need_idf and bracket_on:
        print("Computing IDF for BERTScore (one tokenizer pass) ...", flush=True)
        N = len(items)
        df_per_field: dict[str, dict[int, int]] = {f: {} for f in bracket_fields}
        for (text, sp) in finals:
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=not use_chat)
            ids_i = enc["input_ids"]
            offs_i = enc["offset_mapping"]
            for f in bracket_fields:
                s, e = sp.get(f, (0, 0))
                if e <= s:
                    continue
                seen = set()
                for k, (a, b) in enumerate(offs_i):
                    if a == 0 and b == 0:
                        continue
                    if b > s and a < e:
                        seen.add(int(ids_i[k]))
                for tid in seen:
                    df_per_field[f][tid] = df_per_field[f].get(tid, 0) + 1
        vocab = getattr(model.config, "vocab_size", None) \
            or model.config.text_config.vocab_size  # multimodal (e.g. Gemma3) keeps it under text_config
        idf_dtype = torch.float32
        for f in bracket_fields:
            # BERTScore (Zhang et al. 2019) IDF, matching the official bert_score library:
            #   idf(w) = log((N+1)/(df(w)+1)),  unseen tokens -> log(N+1) (the max).
            # df(w) = # stimuli whose field-f span contains w. NO sklearn-style trailing +1
            # (a constant shift would bias the normalized weighting toward uniform).
            t = torch.full((vocab,), math.log(N + 1.0), dtype=idf_dtype, device=device)
            for tid, c in df_per_field[f].items():
                t[tid] = math.log((N + 1) / (c + 1))
            idf_lookup[f] = t
            seen_n = len(df_per_field[f])
            pos = t[t > 0]
            print(f"  IDF[{f}]: {seen_n} unique token ids, "
                  f"min/median/max IDF = {pos.min().item():.2f} / "
                  f"{pos.median().item():.2f} / {pos.max().item():.2f}", flush=True)

    def to_np(t):
        # fp32, not fp16: some models (e.g. Gemma) have massive deep-layer activations that
        # overflow fp16's 65504 max -> inf -> NaN in downstream CKA/cosine. Storage stays small.
        return t.detach().to(torch.float32).cpu().numpy()

    # Optional constrained answer: token ids for each option string (+ leading-space variant), so the
    # answer argmax is taken over only these instead of the full vocab.
    answer_opts = None
    if args.answer_tokens and not args.no_answer:
        answer_opts = []                       # list of (token_id, label)
        for t in args.answer_tokens:
            for variant in (t, " " + t):
                for tid in tok.encode(variant, add_special_tokens=False):
                    if tid not in {i for i, _ in answer_opts}:
                        answer_opts.append((tid, t))
        print(f"  constrained answer over: {[(tok.convert_ids_to_tokens(i), l) for i, l in answer_opts]}", flush=True)

    t0 = time.time()
    for start in range(0, len(items), args.batch_size):
        chunk = items[start : start + args.batch_size]
        texts = [finals[start + k][0] for k in range(len(chunk))]
        spans = [finals[start + k][1] for k in range(len(chunk))]
        enc = tok(texts, padding=True, truncation=bool(args.max_length), max_length=args.max_length,
                  return_tensors="pt", add_special_tokens=not use_chat,
                  return_offsets_mapping=bracket_on)
        offsets = enc.pop("offset_mapping").to(device) if bracket_on else None
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        B, T = input_ids.shape

        position_ids = attn.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(attn == 0, 1)
        # logits_to_keep=1 keeps only the last position (answer). Span-entropy needs the
        # next-token distribution at every captured span position, so keep all positions there.
        ltk = 0 if (save_span and args.span_entropy) else 1
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, position_ids=position_ids,
                        output_hidden_states=True, use_cache=False, logits_to_keep=ltk)
        hs = out.hidden_states
        rows = torch.arange(B, device=device)
        last_idx = last_token_indices(attn)
        mask_f = attn.unsqueeze(-1).to(hs[layers[0]].dtype)
        denom = mask_f.sum(dim=1).clamp(min=1)

        if "all" in poolings:
            flat_mask = attn.reshape(-1).bool()
            sel = flat_mask.nonzero(as_tuple=False).squeeze(-1)
            sel_row = (sel // T).tolist()
            within = (attn.cumsum(dim=1) - 1).reshape(-1)[sel].tolist()
            sel_ids = input_ids.reshape(-1)[sel].tolist()
            sel_str = tok.convert_ids_to_tokens(sel_ids)
            for r, pos, tid, tstr in zip(sel_row, within, sel_ids, sel_str):
                tok_index_rows.append({"row": start + r, "prompt_id": chunk[r]["id"],
                                       "token_pos": int(pos), "token_id": int(tid), "token_str": tstr})

        for l in layers:
            h = hs[l]
            if "last" in pooled:
                pooled["last"][l].append(to_np(h[rows, last_idx]))
            if "whole" in pooled:
                pooled["whole"][l].append(to_np((h * mask_f).sum(dim=1) / denom))
            if "all" in poolings:
                tokens_all[l].append(to_np(h.reshape(B * T, h.shape[-1])[sel]))

        if bracket_on:
            nonspecial = ~((offsets[:, :, 0] == 0) & (offsets[:, :, 1] == 0))
            # Precompute all bracket fmasks once per batch; both bracket pooling and
            # BERTScore-pair F1 need them, and we want to slice across two fields at a time.
            field_fmask: dict[str, "torch.Tensor"] = {}
            field_cnt: dict[str, "torch.Tensor"] = {}
            for field in bracket_fields:
                fmask = torch.zeros(B, T, dtype=torch.bool, device=device)
                for b in range(B):
                    s, e = spans[b].get(field, (0, 0))
                    if e > s:
                        o = offsets[b]
                        fmask[b] = (o[:, 1] > s) & (o[:, 0] < e)
                fmask &= attn.bool() & nonspecial
                field_fmask[field] = fmask
                field_cnt[field] = fmask.sum(dim=1)

            if save_span:
                # per-token capture: union of bracket spans + optional last-K prompt tokens.
                span_sel = torch.zeros(B, T, dtype=torch.bool, device=device)
                for f in bracket_fields:
                    span_sel |= field_fmask[f]
                if args.last_prompt_tokens > 0:
                    for b in range(B):
                        att = attn[b].nonzero(as_tuple=False).squeeze(-1)
                        span_sel[b, att[-args.last_prompt_tokens:]] = True
                span_sel &= attn.bool()
                flat = span_sel.reshape(-1).nonzero(as_tuple=False).squeeze(-1)   # ascending
                bb = (flat // T).tolist(); pp = (flat % T).tolist()
                ids = input_ids.reshape(-1)[flat].tolist()
                strs = tok.convert_ids_to_tokens(ids)
                # R2 target: next-token entropy H = -sum_v p log p (nats) at each captured position,
                # from the full-vocab output distribution. Gather logits at the captured positions
                # first (small), then reduce, so only [n_captured, vocab] hits fp32.
                if args.span_entropy:
                    lg = out.logits.reshape(B * T, out.logits.shape[-1])[flat].float()
                    logp = torch.log_softmax(lg, dim=-1)
                    ent = (-(logp.exp() * logp).sum(dim=-1)).cpu().tolist()
                last_lab = f"last{args.last_prompt_tokens}"
                ctr: dict = {}
                for i, (b, pos, tid, tstr) in enumerate(zip(bb, pp, ids, strs)):
                    lab = next((f for f in bracket_fields if bool(field_fmask[f][b, pos])), last_lab)
                    key = (start + b, lab); p_in = ctr.get(key, 0); ctr[key] = p_in + 1
                    r = {"row": span_total + i, "prompt_id": chunk[b]["id"],
                         "span": lab, "pos_in_span": p_in, "token_pos": int(pos),
                         "token_id": int(tid), "token_str": tstr}
                    if args.span_entropy:
                        r["next_tok_entropy"] = float(ent[i])
                    span_index_rows.append(r)
                # gather all layers for the selected tokens, stream to disk as [rows, n_layers, d]
                per_layer = []
                for l in layers:
                    t = hs[l].reshape(B * T, hs[l].shape[-1])[flat]
                    if args.span_fp16:
                        per_layer.append(t.detach().clamp_(-65504, 65504).to(torch.float16).cpu().numpy())
                    else:
                        per_layer.append(to_np(t))
                batch_arr = np.stack(per_layer, axis=1)          # [rows, n_layers, d]
                span_fh.write(np.ascontiguousarray(batch_arr).tobytes())
                span_dim = batch_arr.shape[2]
                span_total += batch_arr.shape[0]

            for field in bracket_fields:
                fmask = field_fmask[field]
                cnt = field_cnt[field]
                hs_dtype = hs[layers[0]].dtype
                if "mean" in bracket_modes:
                    fm = fmask.unsqueeze(-1).to(hs_dtype)
                    fden = fm.sum(dim=1).clamp(min=1)
                if "last" in bracket_modes:
                    last_in_span = T - 1 - fmask.flip(dims=[1]).int().argmax(dim=1)
                if "bertscore" in bracket_modes:
                    idf_w = idf_lookup[field][input_ids] * fmask.to(idf_lookup[field].dtype)
                    idf_den = idf_w.sum(dim=1, keepdim=True).clamp(min=1e-6)
                    fallback = (idf_w.sum(dim=1) == 0) & (cnt > 0)
                    if fallback.any():
                        uniform = fmask.to(idf_lookup[field].dtype)
                        idf_w = torch.where(fallback.unsqueeze(1), uniform, idf_w)
                        idf_den = idf_w.sum(dim=1, keepdim=True).clamp(min=1e-6)
                    idf_w_norm = (idf_w / idf_den).unsqueeze(-1).to(hs_dtype)
                for l in layers:
                    if "mean" in bracket_modes:
                        v = (hs[l] * fm).sum(dim=1) / fden
                        v[cnt == 0] = float("nan")
                        bracket_acc["mean"][field][l].append(to_np(v))
                    if "last" in bracket_modes:
                        vL = hs[l][rows, last_in_span].clone()
                        vL[cnt == 0] = float("nan")
                        bracket_acc["last"][field][l].append(to_np(vL))
                    if "bertscore" in bracket_modes:
                        vB = (hs[l] * idf_w_norm).sum(dim=1)
                        vB[cnt == 0] = float("nan")
                        bracket_acc["bertscore"][field][l].append(to_np(vB))
                bracket_cnt[field].extend(cnt.tolist())

            # ---- BERTScore cross-layer F1 ----
            # For each pair (A, B) and each stimulus in the batch, build the L_A × L_B
            # matrix where M[i, j] = F1(A's token vectors at layer i, B's token vectors at layer j).
            # Stored per stimulus so analyses can subset (e.g. by model choice) later.
            if bs_pairs:
                L_count = len(layers)
                # Pre-normalize hidden states for cosine (in fp32 for numerical stability).
                hs_norm = []
                for l in layers:
                    h32 = hs[l].to(torch.float32)
                    hs_norm.append(h32 / h32.norm(dim=-1, keepdim=True).clamp(min=1e-8))
                hs_norm_stack = torch.stack(hs_norm, dim=0)  # [L, B, T, d]
                idf_on = need_idf and not args.bertscore_no_idf
                for (A, B_field) in bs_pairs:
                    cnt_A = field_cnt[A]
                    cnt_B = field_cnt[B_field]
                    for stim in range(B):
                        nA, nB = int(cnt_A[stim]), int(cnt_B[stim])
                        if nA == 0 or nB == 0:
                            nan_lxl = np.full((L_count, L_count), np.nan, dtype=np.float16)
                            bs_acc_F1[(A, B_field)].append(nan_lxl)
                            if args.bertscore_save_pr:
                                bs_acc_P[(A, B_field)].append(nan_lxl)
                                bs_acc_R[(A, B_field)].append(nan_lxl)
                            continue
                        a_pos = field_fmask[A][stim].nonzero(as_tuple=True)[0]
                        b_pos = field_fmask[B_field][stim].nonzero(as_tuple=True)[0]
                        # Slice tokens at all layers: [L, n, d]
                        A_tok = hs_norm_stack[:, stim, a_pos, :]
                        B_tok = hs_norm_stack[:, stim, b_pos, :]
                        # Cross-layer cosine cube: [L_A, L_B, n_A, n_B].
                        S = torch.einsum("iad,jbd->ijab", A_tok, B_tok)
                        if idf_on:
                            w_A = idf_lookup[A][input_ids[stim, a_pos]].to(S.dtype)
                            w_B = idf_lookup[B_field][input_ids[stim, b_pos]].to(S.dtype)
                            if w_A.sum() == 0:
                                w_A = torch.ones_like(w_A)
                            if w_B.sum() == 0:
                                w_B = torch.ones_like(w_B)
                            R_t = (S.max(dim=3).values * w_A).sum(dim=2) / w_A.sum()
                            P_t = (S.max(dim=2).values * w_B).sum(dim=2) / w_B.sum()
                        else:
                            R_t = S.max(dim=3).values.mean(dim=2)
                            P_t = S.max(dim=2).values.mean(dim=2)
                        # Clamp P,R to >=0 before the harmonic mean: token max-cosines can be
                        # negative (esp. cross-layer, where A's and B's spaces differ), and a
                        # raw 2PR/(P+R) explodes when P+R~=0 (a negative match contributes "no
                        # match" -> 0). Bounds F1 to [0,1]; same-layer values are unaffected.
                        R_t = R_t.clamp(min=0.0)
                        P_t = P_t.clamp(min=0.0)
                        F1_t = 2 * P_t * R_t / (P_t + R_t).clamp(min=1e-8)
                        bs_acc_F1[(A, B_field)].append(F1_t.to(torch.float16).cpu().numpy())
                        if args.bertscore_save_pr:
                            bs_acc_P[(A, B_field)].append(P_t.to(torch.float16).cpu().numpy())
                            bs_acc_R[(A, B_field)].append(R_t.to(torch.float16).cpu().numpy())
                # Free the stacked normalized states before next batch.
                del hs_norm, hs_norm_stack

        if not args.no_answer:
            logits_last = out.logits[:, -1, :]
            if answer_opts:                    # argmax over only the option tokens
                opt_ids = torch.tensor([i for i, _ in answer_opts], device=logits_last.device)
                win = logits_last.index_select(1, opt_ids).argmax(dim=-1)
                ans_id = opt_ids[win]
                ans_text = [answer_opts[w][1] for w in win.tolist()]
            else:                              # unrestricted full-vocab argmax
                ans_id = logits_last.argmax(dim=-1)
                ans_text = tok.batch_decode(ans_id.unsqueeze(-1))
            ans_prob = torch.softmax(logits_last.float(), dim=-1)[rows, ans_id]   # full-vocab prob of the pick
            ans_id, ans_prob = ans_id.tolist(), ans_prob.tolist()
        n_tokens = attn.sum(dim=1).tolist()

        for k, item in enumerate(chunk):
            row = {"row": start + k, "prompt_id": item["id"],
                   "prompt_sha1": hashlib.sha1(item["text"].encode("utf-8")).hexdigest()[:12],
                   "n_tokens": int(n_tokens[k])}
            if "sentence_col" in item:
                row["sentence_col"] = item["sentence_col"]
            if not args.no_answer:
                row.update(answer_token_id=int(ans_id[k]), answer_text=ans_text[k], answer_prob=float(ans_prob[k]))
            meta_rows.append(row)
        print(f"  {min(start + args.batch_size, len(items))}/{len(items)} prompts ({time.time() - t0:.1f}s)", flush=True)

    pooled_final = {name: {l: np.concatenate(c, 0) for l, c in d.items()} for name, d in pooled.items()}
    for field in bracket_fields:
        if "mean" in bracket_modes:
            pooled_final[f"bracket_{field}"] = {l: np.concatenate(c, 0) for l, c in bracket_acc["mean"][field].items()}
        if "last" in bracket_modes:
            pooled_final[f"bracket_{field}_last"] = {l: np.concatenate(c, 0) for l, c in bracket_acc["last"][field].items()}
        if "bertscore" in bracket_modes:
            pooled_final[f"bracket_{field}_bertscore"] = {l: np.concatenate(c, 0) for l, c in bracket_acc["bertscore"][field].items()}
    tokens_all_final = ({l: np.concatenate(c, 0) for l, c in tokens_all.items()} if "all" in poolings else None)
    tokens_index_df = pd.DataFrame(tok_index_rows) if "all" in poolings else None

    metadata_df = pd.DataFrame(meta_rows)
    for field in bracket_fields:
        metadata_df[f"n_{field}"] = bracket_cnt[field]

    resolved_set_name = set_name or (Path(template_path).stem if template_path
                                     else Path(args.prompts).stem if args.prompts
                                     else time.strftime("run_%Y%m%d_%H%M%S"))
    run_dir = Path(args.output_dir) / alias / resolved_set_name
    manifest = {
        "model_alias": alias, "hf_id": hf_id, "dtype": args.dtype, "load_4bit": args.load_4bit, "device": str(device),
        "chat_template": use_chat, "thinking": thinking, "max_length": args.max_length,
        "layers": layers, "n_hidden_states": n_states, "pooling": poolings,
        "brackets": bracket_fields, "bracket_pooling": bracket_modes,
        "save_span_tokens": save_span, "last_prompt_tokens": args.last_prompt_tokens if save_span else None,
        "span_dtype": ("fp16" if args.span_fp16 else "fp32") if save_span else None,
        "template": template_path, "stimuli": args.stimuli if template_path else None,
        "sentences": args.sentences, "sentence_slot": args.sentence_slot if carrier_mode else None,
        "counterbalance": args.counterbalance, "pass_label": pass_label or None,
        "n_prompts": len(items), "record_answer": not args.no_answer,
        "answer_tokens": args.answer_tokens,
        "transformers": transformers.__version__, "torch": torch.__version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if bs_pairs:
        manifest["bertscore"] = {
            "pairs": [[a, b] for a, b in bs_pairs],
            "layers": layers,
            "idf": (need_idf and not args.bertscore_no_idf),
            "save_pr": args.bertscore_save_pr,
        }
    embeddings_io.save_run(run_dir, pooled=pooled_final, metadata=metadata_df, manifest=manifest,
                           tokens_all=tokens_all_final, tokens_index=tokens_index_df)

    # Case A per-token span capture (curvature): span_tokens.bin was streamed per batch as
    # [rows, n_layers, d] (peak RAM ~ one batch). Close it, write the index + shape metadata.
    if save_span:
        span_fh.close()
        span_index_df = pd.DataFrame(span_index_rows)
        span_index_df.to_parquet(span_run_dir / "span_tokens_index.parquet")
        _sd = "fp16" if args.span_fp16 else "fp32"
        span_meta = {"file": "span_tokens.bin", "layout": "[rows, n_layers, d]", "dtype": _sd,
                     "rows": int(span_total), "n_layers": len(layers), "d": int(span_dim or 0),
                     "has_entropy": bool(args.span_entropy)}
        (span_run_dir / "span_tokens_meta.json").write_text(json.dumps(span_meta, indent=2))
        print(f"  span tokens: {span_total} rows x {len(layers)} layers x {span_dim} d ({_sd}) "
              f"({dict(span_index_df['span'].value_counts())}) -> span_tokens.bin (memmap)", flush=True)

    # BERTScore pairs: separate safetensors file with [N, L_A, L_B] per pair.
    if bs_pairs:
        from safetensors.numpy import save_file as save_st
        bs_tensors: dict[str, np.ndarray] = {}
        for pair in bs_pairs:
            a, b = pair
            bs_tensors[f"{a}__{b}__F1"] = np.stack(bs_acc_F1[pair], axis=0)
            if args.bertscore_save_pr:
                bs_tensors[f"{a}__{b}__P"] = np.stack(bs_acc_P[pair], axis=0)
                bs_tensors[f"{a}__{b}__R"] = np.stack(bs_acc_R[pair], axis=0)
        bs_path = run_dir / "bertscore_pairs.safetensors"
        save_st({k: np.ascontiguousarray(v) for k, v in bs_tensors.items()}, str(bs_path))
        print(f"  bertscore_pairs.safetensors: {len(bs_pairs)} pairs × "
              f"{'F1+P+R' if args.bertscore_save_pr else 'F1 only'}, "
              f"{next(iter(bs_tensors.values())).shape} per key (fp16)")

    print(f"\nWrote {run_dir}")
    for name, d in pooled_final.items():
        any_l = next(iter(d.values()))
        print(f"  pooled_{name}.safetensors: {len(d)} layers x {any_l.shape} (fp16)")
    if tokens_all_final is not None:
        any_l = next(iter(tokens_all_final.values()))
        print(f"  tokens_all.safetensors:   {len(tokens_all_final)} layers x {any_l.shape}")


def main() -> None:
    args = parse_args()
    poolings = list(dict.fromkeys(args.pooling))
    bracket_modes = list(dict.fromkeys(args.bracket_pooling))

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pandas as pd

    alias, hf_id, chat_default = resolve_model(args.model, args.models_config)
    use_chat = chat_default and not args.raw
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    prec = f"4-bit nf4 (compute {args.dtype})" if args.load_4bit else args.dtype
    print(f"Loading {hf_id} ({prec}, device={args.device}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(hf_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if use_chat and tok.chat_template is None:
        print("  (no chat template on this model -> falling back to --raw)")
        use_chat = False

    dtype_kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    load_kwargs = {dtype_kw: dtype}
    if args.attn_impl:
        load_kwargs["attn_implementation"] = args.attn_impl
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        load_kwargs["device_map"] = "auto"          # 4-bit requires device_map dispatch
    elif args.device == "auto":
        load_kwargs["device_map"] = "auto"
    if args.max_mem_per_gpu and load_kwargs.get("device_map") == "auto":
        n = torch.cuda.device_count()
        load_kwargs["max_memory"] = {i: args.max_mem_per_gpu for i in range(n)}
        print(f"  max_memory cap: {args.max_mem_per_gpu}/GPU across {n} GPUs (overflow -> CPU offload)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
    if args.device != "auto" and not args.load_4bit:
        model.to(args.device)
    model.eval()
    device = next(model.parameters()).device

    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", None) \
        or getattr(getattr(cfg, "text_config", None), "num_hidden_layers", None)  # multimodal (e.g. Gemma3)
    n_states = n_layers + 1
    layers = resolve_layers(args.layers, n_states)

    # Decide passes. Without --counterbalance: one pass with the original template.
    # With --counterbalance A B: two passes. The second uses a temp template where
    # {A} and {B} have been swapped; both passes write under <set_name>_{fwd,rev}/.
    passes: list[tuple[str | None, str | None, str]] = []   # (template_path, set_name, pass_label)
    if args.counterbalance:
        a, b = args.counterbalance
        if not args.template:
            raise SystemExit("--counterbalance requires --template.")
        base = args.set_name or Path(args.template).stem
        original_text = Path(args.template).read_text(encoding="utf-8")
        swapped_text = swap_placeholders(original_text, a, b)
        if swapped_text == original_text:
            raise SystemExit(f"--counterbalance {a} {b}: neither {{{a}}} nor {{{b}}} found in the template.")
        tmp_dir = Path(tempfile.mkdtemp(prefix="extract_swap_"))
        swapped_path = tmp_dir / f"{Path(args.template).stem}__swap_{a}_{b}.txt"
        swapped_path.write_text(swapped_text, encoding="utf-8")
        passes.append((args.template,        f"{base}_fwd", "fwd"))
        passes.append((str(swapped_path),    f"{base}_rev", f"rev (swapped {{{a}}} ↔ {{{b}}})"))
        print(f"--counterbalance enabled: 2 passes, swapping {{{a}}} ↔ {{{b}}}. "
              f"Swapped template at {swapped_path}.")
    else:
        passes.append((args.template, args.set_name, ""))

    for template_path, set_name, label in passes:
        run_pass(
            args,
            template_path=template_path,
            set_name=set_name,
            pass_label=label,
            torch=torch, transformers=transformers, pd=pd,
            model=model, tok=tok, device=device,
            alias=alias, hf_id=hf_id, use_chat=use_chat,
            n_states=n_states, layers=layers,
            poolings=poolings, bracket_modes=bracket_modes,
        )


if __name__ == "__main__":
    main()

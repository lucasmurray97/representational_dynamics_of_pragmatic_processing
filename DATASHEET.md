# Datasheet: Spanish conversational-implicature stimuli

A datasheet for the stimulus set in `data/stimuli/items_648.xlsx` and the prompt templates in
`data/prompts/`, following the *Datasheets for Datasets* structure.

## Motivation

The stimuli probe whether a language model recovers the **intended (pragmatic) meaning** of an
indirect answer, as opposed to its literal content. Each item stages a two-party exchange in which
speaker *A* asks a polar question and speaker *B* replies indirectly; the model must judge which of
two candidate sentences better captures what *B* meant. The set was built to support both a
behavioural measure (pragmatic uptake) and a representational analysis of the model's per-layer
token trajectories over four content elements.

## Composition

- **648 items**, each a self-contained exchange, in Spanish.
- Four content elements per item, used as the analysis spans:
  - **QUD / QUD_polar** — the question under discussion posed by *A* (polar form used in the task).
  - **U** — *B*'s indirect response (the source of the implicature).
  - **I** — the implicature reading (the pragmatically intended answer; an answer option).
  - **PU** — the literal paraphrase of *B*'s response (an answer option).
- Additional columns: `Item` (id), `categoria`, `verb`, `clau` (construction metadata), `U2`
  (an alternate response), and distractor/control fields `CT1`–`CT3`, `NR`.
- No personal data. Names in the items (e.g. "Juan") are generic placeholders, not real individuals.

## Task format

The prompt templates `data/prompts/repr_conc_polar_{0..4}shot.txt` present the exchange and two
answer options (Oración 1 = **I**, Oración 2 = **PU**), asking the model to reply with `1` or `2`.
The 1–4-shot templates prepend that many worked demonstrations (fixed, drawn from held-out items).
To control for answer-position bias, extraction is **counterbalanced**: each item is presented once
with the options in each order (`fwd` / `rev`), and a response counts as pragmatic only if the model
selects the implicature option in both orders.

## Collection & construction

The items were authored as controlled minimal exchanges spanning several implicature categories
(`categoria`), each with a matched literal paraphrase (PU) so that the pragmatic and literal readings
differ in meaning but not in surface plausibility. The stimuli are synthetic authored materials; they
were not scraped from user data.

## Uses

Intended for research on pragmatic competence and representational analysis in language models.
Because the materials are Spanish and synthetic, they are not a general benchmark of language ability
and should not be treated as one. They are not suitable for training data-contamination checks or for
any use involving real individuals.

## Distribution & license

Released with this repository under the MIT License (see `LICENSE`). The behavioural responses in
`data/responses/` were produced by running publicly available Hugging Face checkpoints (listed in
`models.csv`) on these stimuli; each model remains under its own upstream license.

# Notebook 04 — Benchmark Data

External data files for **Notebook 04 (Plain LLM vs RLM Comparison)**.

These files contain a synthetic corporate annual report used as the benchmark
context, the poisoned variant with injected adversarial instructions, and the
ground-truth questions/answers used for scoring.

## Files

| File | Description |
|---|---|
| `long_report.txt` | Clean corporate annual report (~30 000 words, 15 sections) with facts embedded in natural-language prose. Numbers are written as words, dates are written conversationally. Includes near-miss distractor figures that could be confused with the correct answers. Uses `SECTION:` separators compatible with the RLM splitting pattern. |
| `injected_report.txt` | Same report with three **data-corruption-style** injection payloads: an erratum (Financial Performance), a correction notice (Leadership), and a security addendum (Security & Reliability). These mimic legitimate editorial elements rather than crude "IGNORE ALL" attacks. |
| `questions.json` | Ground-truth question set (Q1–Q7): each question requires information from 2–4 sections. Answers are expressed in prose. Each question has multiple `components` with per-component keywords and source sections, enabling exactness and completeness scoring. |

## Benchmark design principles

- **30 000+ words** across **15 sections** to stress long-context comprehension.
- Each question requires synthesising facts from **2–4 sections** (not just one).
- Numerous **near-miss distractors**: similar numbers, partial figures, and
  related-but-wrong data points are scattered throughout to test precision.
- All answers are expressed in **natural-language prose** (no machine-friendly
  markers or unique identifiers) — models must actually comprehend the text.
- Evaluation measures both **exactness** (are the specific values correct?) and
  **completeness** (were all required components of the answer found?).

## Section separator format

Both report files use the block-format section separator established in
Notebook 03:

```
============================================================
SECTION: <Name>
============================================================
```

Lines of exactly 60 `=` characters with `SECTION: <name>` on the middle line.

## Questions format

Each question in `questions.json` has the following structure:

```json
{
  "Q1": {
    "question": "...",
    "answer": "...",
    "components": [
      {
        "label": "component_name",
        "description": "What this component tests",
        "keywords": ["keyword1", "keyword2"],
        "source_sections": ["Section A", "Section B"]
      }
    ],
    "distractors_note": "Description of near-miss distractors"
  }
}
```

## Updating the benchmark

To modify the report content, questions, or injections, edit the relevant file
directly and re-run Notebook 04 to verify results. The notebook loads these
files at the start and does **not** generate them inline.

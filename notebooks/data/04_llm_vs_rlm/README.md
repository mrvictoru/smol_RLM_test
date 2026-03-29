# Notebook 04 — Benchmark Data

External data files for **Notebook 04 (Plain LLM vs RLM Comparison)**.

These files contain the corporate annual report used as the benchmark context,
the poisoned variant with injected adversarial instructions, and the
ground-truth questions/answers used for scoring.

## Files

| File | Description |
|---|---|
| `long_report.txt` | Clean corporate annual report (~2,000 words, 6 sections) with facts embedded in natural-language prose. Uses `SECTION:` separators compatible with the RLM splitting pattern. |
| `injected_report.txt` | Same report with two adversarial prompt-injection payloads inserted into the *Financial Performance* and *Operations & Infrastructure* sections. |
| `questions.json` | Ground-truth question set (Q1–Q5): question text, expected answer, keyword list for scoring, and source section name. |

## Section separator format

Both report files use the block-format section separator established in
Notebook 03:

```
============================================================
SECTION: <Name>
============================================================
```

Lines of exactly 60 `=` characters with `SECTION: <name>` on the middle line.

## Updating the benchmark

To modify the report content, questions, or injections, edit the relevant file
directly and re-run Notebook 04 to verify results. The notebook loads these
files at the start and does **not** generate them inline.

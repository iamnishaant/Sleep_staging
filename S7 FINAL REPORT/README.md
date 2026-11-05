# S7 final report (release v2)

LaTeX project in the university template's structure (`S7 Report Template.zip`), filled for release v2.

| File | What |
|---|---|
| `mythesis.tex` | Main file: title page, bonafide, contents, abbreviations, abstract, chapters, references |
| `chapters/*.tex` | Abstract, abbreviations, Chapters 1–5, Appendix A (code and reproduction) |
| `mybib.bib` | 38 references (IEEE style); entries noted "check" need authors/venue confirmed before submission |
| `images/` | 31 figures: 6 new architecture diagrams (`A1`–`A6`), 19 v2 result figures from `figures/output_v2/`, 6 v1-history figures (`v1_*`), template logos |
| `make_architecture_figures.py` | Regenerates `A1`–`A6` from the v2 code facts |

## Compile

Overleaf: upload `S7_FINAL_REPORT_overleaf.zip`, set the main document to `mythesis.tex`, compiler pdfLaTeX.
Locally: `pdflatex mythesis && bibtex mythesis && pdflatex mythesis && pdflatex mythesis`.

## Still to fill in by hand

Names and register numbers (title page and bonafide), guide / coordinator / reviewer, acknowledgements.

## Rules followed

- Every number is from `CURRENT_RELEASE.md` and the result files listed in Appendix A.4.
- v1 staging numbers on the original stored labels appear only in the historical distillation section, labelled.
- `distillation/figures/fig_blockwise_architecture.png` (v1, hand-drawn) is **not** used: it shows the old 121 K
  student, an NLI verifier and longitudinal reasoning that were never built. Its v2 successor is
  `A0_blockwise_architecture_v2` (`make_blockwise_v2.py`, also saved as `distillation/figures/fig_blockwise_architecture_v2.png`).
- Edge simulation: only the measured L1 container results are claimed; the report step in the container, ARM64
  and the Pi are stated as pending.

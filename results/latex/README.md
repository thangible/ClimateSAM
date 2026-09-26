# LaTeX chapter for Section 4.3

| File | Content | Goes into |
|---|---|---|
| `approach.tex` | evaluation protocol (the empty Section 3.6), mask-prompt generator, training modes, decoder adaptation / prompt-robust decoder | Chapter 3 |
| `section_4_3.tex` | Section 4.3 with 4.3.1 Prompt Analysis, 4.3.2 SAM with CG-Net as Prompter, 4.3.3 Generator from SAM Inputs, error analysis, decoder adaptation, evaluation-only improvements, static prompts, second checkpoint, examples, summary | Chapter 4 |
| `discussion.tex` | why SAM cannot improve on its prompter, why a small generator wins, prompt design, implications, limitations, future work | Discussion |
| `appendix.tex` | complete tables, second example, issues found in the original code | Appendix |
| `main.tex` | standalone build of all parts, for checking (`tectonic main.tex` or `pdflatex` twice) | — |

In the thesis:

1. Packages: `amsmath`, `amssymb`, `graphicx`, `booktabs`, `adjustbox` (tables shrink to the text width only if needed).
2. Define where the results folder is, relative to the main `.tex` file, e.g. `\newcommand{\ResultsDir}{results}`.
3. `\input{results/latex/section_4_3}` etc. at the right places. Tables are `\input` from `results/tables/*.tex`
   and figures come from `results/figures/*.pdf`, so re-running `make_report.py` / `figures.py` updates the chapter.
4. References to existing thesis objects (Table 4.4, 4.12, 4.13, Figure 3.9, Eq. 3.4, Section 2.2.2, Table 2.1) are
   written as fixed numbers; replace them by `\ref{...}` with your labels. Labels defined here start with `sec:`,
   `tab:`, `fig:`, `eq:` and do not clash with the generated table labels.

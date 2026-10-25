# Paper

| File | What it is |
|---|---|
| [`preprint.pdf`](preprint.pdf) | The paper, ready to read (8 pages) |
| [`latex/`](latex/) | LaTeX source: `main.tex`, `refs.bib`, `arxiv.sty`, vector figures |
| `latex/main.bbl` | Compiled bibliography, needed by arXiv (arXiv does not run BibTeX) |

To rebuild: `cd latex && latexmk -pdf main.tex`.
To submit to arXiv: zip `main.tex`, `main.bbl`, `refs.bib`, `arxiv.sty` and `figures/` (not `main.pdf`) and upload the zip.

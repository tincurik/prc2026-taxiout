# Paper: JOAS-format article

`main.pdf` is the model written up as a research article in the format of the
[Journal of Open Aviation Science](https://journals.open.tudelft.nl/joas) (JOAS): its LaTeX class (`extra/joas.cls`, from the
journal's template repository, LPPL 1.3c), one `main.tex`, `reference.bib` and a `figures/` folder, as the template requires.
It covers the same ground as `docs/`: the task and data, the label and its record mechanisms, the model, its validation and
results, and its limits.

```
main.tex          the article (single file, as JOAS asks)
reference.bib     references (biblatex + biber)
figures/          vector PDF figures, drawn by make_figures.py
make_figures.py   draws figures/ from ../writeup/data/*.csv with the write-up's drawing code, restyled for print
build.sh          redraws the figures and builds main.pdf (pdflatex, biber, pdflatex twice)
extra/            the JOAS class and the journal logo the class prints in the header, both as shipped in the journal's template
                  repository; the logo is the journal's mark, not covered by this repository's GPLv3
```

Build (needs a TeX distribution with pdflatex and biber, and Python with matplotlib, pandas and numpy):

```
bash build.sh            # PY=<python> selects the interpreter
```

Each number in the paper matches a table in `docs/` or a CSV in `writeup/data/`.

The paper is a preprint and has not been submitted anywhere; it is set in the JOAS preprint layout for formatting only (a footnote on page one says so).
The author is Tin Josip Curik (`main.tex`, `CITATION.cff`); the authorship and AI-assistance statement is in the paper (section "Authorship and AI assistance") and in the
[README](../README.md#authorship-and-ai-assistance).

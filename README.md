# Enterprise Document Intelligence — Vol.1 notebooks

Runnable, **code-only** companions to the *Enterprise Document Intelligence*
article series on Towards Data Science. Each notebook runs one article's pipeline
end to end. The explanations, the diagrams and the *why* live in the article;
here you just run the code.

**Read the series:** [all articles by Angela Shi](https://towardsdatascience.com/author/angela-shi/)

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -e .
cp .env.example .env          # then fill in API_KEY / BASE_URL / MODEL_CHAT
```

The notebooks import the bundled `lib/` package (a slice of the production code)
and read sample documents from `data/`. Open any notebook and run all cells.

## Notebooks

- `notebooks/02_2_rerankers.ipynb`
- `notebooks/05_5_easyocr_parsing.ipynb`
- `notebooks/06_A_thesis.ipynb`
- `notebooks/06_B_extraction.ipynb`
- `notebooks/06_C_dispatch.ipynb`
- `notebooks/06_question_parsing.ipynb`
- `notebooks/07_A_filtering.ipynb`
- `notebooks/07_B_detection.ipynb`
- `notebooks/07_C_arbiter.ipynb`
- `notebooks/07_retrieval.ipynb`

Every section in a notebook links back to the matching section of the published
article. To understand a step, read the article.

## Support

These notebooks ship a runnable slice of the code. **For the complete code**
(every brick in production-shaped form, the dispatcher, the schemas), get in
touch on Ko-fi: https://ko-fi.com/angelashi

## License

Proprietary. Personal, educational use only (read it, run it locally alongside
the articles). No redistribution or commercial use without written permission.
See `LICENSE`.

**Absolute Rules (must follow)**

1. **No assumptions. No invented content.** Everything must be supported by repository evidence (file path + function/class/line range if possible).
2. **Review before writing.** Traverse all subfolders and files (backend, frontend, models, data, scripts, configs, Docker, tests, notebooks). Parse code, configs, README, `.env.example`, `requirements`, and any docs.
3. **Evidence tagging.** For every subsection, include a small “Evidence” block with citations like:

   `- backend/app/services/data_service.py#L42-L118: get_history_df()`

   `- models/informer_model.py#L10-L85: SimpleInformer`
4. **Exact order and coverage.** Follow the presentation index below, filling every topic and subtopic. If repo lacks something, write `MISSING` and add a precise TODO.
5. **Repository-first.** Do not use web search or external sources unless the repo explicitly requires it; if used, label it as `External Reference` and keep it minimal.
6. **Clarity + completeness.** Expand acronyms on first use: e.g., RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence), DQN (Deep Q-Network), SHAP (Shapley Additive Explanations), OHLCV (Open, High, Low, Close, Volume), PE (Price-to-Earnings), MAE (Mean Absolute Error), MSE (Mean Squared Error), RMSE (Root Mean Squared Error), MAPE (Mean Absolute Percentage Error).
7. **Grounded to your code.** Where you mention a model, feature, dataset, metric, or endpoint, show **exact code anchors** (file + function/class + line range if possible) and any default hyperparameters found in code/config.

**Deliverable**

* Create a single Markdown file named:

  `TRAE_STOCK/docs/Full_presentation.md`

  (Create `docs/` if it doesn’t exist.)
* Use the exact section numbering and subheadings below.
* Prefer concise, readable paragraphs, plus tables for lists (features, indicators, metrics), and bullet points for steps.
* Where charts are mentioned, include **a “How this is generated” note** that points to the script/component that creates it (or write a TODO if missing).

## 📌 Extra Formatting & Utilities

**At the top of the document, also include:**

* **Project Snapshot Table** :
* Frameworks: FastAPI, React, Tailwind, Docker, Celery, Redis, PostgreSQL
* Models present: XGBoost, Informer/Transformer, DQN
* Data sources: yfinance, Angel One SmartAPI, News scrapers/NewsAPI
* Explainability: SHAP
* Metrics: MAE/MSE/RMSE/MAPE (+ others if used)
* Key endpoints: list FastAPI routes with brief purpose and file anchors

**Library Glossary** (expand acronyms and give one-liners).

**Reproducibility Appendix (optional)**

* If found in repo: environment creation commands, `.env.example` variables, start commands (backend, frontend), Docker/Nginx notes. Otherwise mark `MISSING`.

## 📝 Execution Steps (what you must do)

1. **Scan** all files and build an internal index (paths, functions, classes, docstrings, comments, configs).
2. **Extract** data sources, pipelines, feature engineering, model definitions, training/eval code, endpoints, and UI components.
3. **Verify** every statement with a file/line anchor.
4. **Write** `TRAE_STOCK/docs/Full_presentation.md` following the sections and evidence blocks.
5. **Self-check** for any unverified statement; either add an anchor or replace with `MISSING` + TODO.
6. **Save** the file. Do not output anything else.

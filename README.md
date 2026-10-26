# AI-WM Lab — Unified Biomedical Database Explorer

A **single Streamlit application** containing the original MIMIC-III and VitalDB explorers, with a new database-selection homepage.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a **new GitHub repository** (for example `aiwm-biomedical-database-explorer`).
2. Upload the **contents** of this folder to the repository root, retaining the directory structure. GitHub web uploads have file-size restrictions; the bundled MIMIC-III deployment ZIP is approximately 59 MB and should be pushed with Git rather than uploaded through the browser if the browser rejects it.
3. In Streamlit Community Cloud, select **Create app**, choose the new repository, and set **Main file path** to `app.py`.
4. Deploy and verify the MIMIC-III overview, record queries, VitalDB metadata filters, on-demand signal retrieval, and downloads.
5. Keep the existing two deployed applications unchanged until verification is complete.

## Directory structure

- `app.py`: unified homepage and navigation
- `mimic3/src/13_app.py`: original MIMIC-III explorer (page config removed)
- `mimic3/mimic3_deploy_db.zip`: original compressed DuckDB deployment database
- `vitaldb/src/13_explorer_ui.py`: original VitalDB explorer (page config removed)
- `vitaldb/src/07_build_remote_signal_retrieval_engine.py`: remote retrieval engine
- `vitaldb/src/12_build_cohort_download_export_engine.py`: export engine
- `vitaldb/metadata/`, `vitaldb/results/stage10/`: original metadata and validity policy

## Changes made

Only the two child-app `st.set_page_config(...)` calls were removed. The launcher now sets page configuration once and uses native `st.navigation` / `st.switch_page` to route users within the same Streamlit website. The original data processing, SQL, filters, and export logic remain unchanged.

## Scientific and operational notes

MIMIC-III waveform presence is metadata-based; sample-level quality and exact continuous signal availability are not established by the existing application. VitalDB remote signal retrieval can fetch a complete remote track before local clipping, and is triggered explicitly by the user. Large on-demand downloads can consume Streamlit Community Cloud memory and bandwidth. Test deployed workflows before retiring older apps.

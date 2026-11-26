# CreditWise - Loan Approval Predictor (Streamlit)

A mobile-first Streamlit front end for the `credit_wise.ipynb` loan approval project.
The notebook is not modified: `app.py` repeats its pipeline (imputation, encoding, squared
Credit_Score / DTI_Ratio, 80/20 split, StandardScaler) and trains the same three models on start-up.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this whole folder to a GitHub repo (keep `loan_approval_data.csv`, `assets/` and `.streamlit/`).
2. On share.streamlit.io choose the repo, branch and `app.py`, then Deploy.

## Make it yours
- Name and links: edit `AUTHOR`, `GITHUB_URL`, `LINKEDIN_URL` at the top of `app.py`.
- Background photo: drop a file named `bg.jpg` (or `.png` / `.webp`) into `assets/`. It replaces the generated skyline automatically.
- Colours: change the CSS variables at the top of the `CSS` string in `app.py` (white theme: navy text, gold and teal accents).

## Files
```
app.py                  the app
loan_approval_data.csv  dataset (same file as the notebook)
requirements.txt
assets/bg_city.svg      light background skyline
assets/hero_dial.svg    hero illustration
.streamlit/config.toml  light theme + minimal toolbar
```

# CreditWise - Loan Approval Predictor (Streamlit)

A mobile-first Streamlit front end for the `credit_wise.ipynb` loan approval project.
The notebook is not modified: `app.py` repeats its pipeline (imputation, encoding, squared
Credit_Score / DTI_Ratio, 80/20 split, StandardScaler) and trains the same three models on start-up.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```



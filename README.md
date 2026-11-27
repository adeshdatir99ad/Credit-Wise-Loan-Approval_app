# CreditWise - Loan Approval Predictor (Streamlit)

A mobile-first Streamlit front end for the `credit_wise.ipynb` loan approval project.
The notebook is not modified: `app.py` repeats its pipeline (imputation, encoding, squared
Credit_Score / DTI_Ratio, 80/20 split, StandardScaler) and trains the same three models on start-up.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

<img width="1366" height="642" alt="Screenshot 2026-11-26 225615" src="https://github.com/user-attachments/assets/4d8ecab0-fef0-4c2f-8a97-f855ed5a4665" />

<img width="1366" height="642" alt="Screenshot 2026-11-26 225640" src="https://github.com/user-attachments/assets/1b4dc289-4947-4824-80ed-37384c7c343e" />

<img width="1366" height="641" alt="Screenshot 2026-11-26 225735" src="https://github.com/user-attachments/assets/2a0124a1-03a0-4255-92a7-48af892818b9" />

<img width="1366" height="640" alt="Screenshot 2026-11-26 225757" src="https://github.com/user-attachments/assets/466cfe8e-1943-422a-ac62-25e552ec1ef3" />

"""
CreditWise - loan approval predictor (Streamlit front end)

The model code below repeats the exact steps of credit_wise.ipynb:
  impute -> drop Applicant_ID -> label/one-hot encode -> square Credit_Score & DTI_Ratio
  -> 80/20 split (random_state=42) -> StandardScaler -> Naive Bayes / Logistic Regression / KNN.
The notebook itself is untouched; this file only wraps it in an interface.
"""
import base64
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# ------------------------------- settings ----------------------------------
AUTHOR = "Adesh Datir"
GITHUB_URL = "https://github.com/adeshdatir99ad/Credit-Wise-Loan-Approval_app"
LINKEDIN_URL = "https://www.linkedin.com/in/adeshdatir/"
FINAL_MODEL = "Naive Bayes"   # the model the notebook settled on

st.set_page_config(page_title="CreditWise | Loan approval predictor", page_icon="🏦",
                   layout="centered", initial_sidebar_state="collapsed")

BASE = Path(__file__).parent
DATA_PATH = BASE / "loan_approval_data.csv"
ASSETS = BASE / "assets"
CAT_COLS = ["Employment_Status", "Marital_Status", "Loan_Purpose",
            "Property_Area", "Gender", "Employer_Category"]
RING_C = 2 * math.pi * 54  # circumference of the result ring


# --------------------------------------------------------------------------- small helpers
def html(markup: str) -> None:
    """Render raw HTML. Blank lines and indentation are removed so Markdown never breaks the block."""
    st.markdown("\n".join(ln.strip() for ln in markup.splitlines() if ln.strip()),
                unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def asset_uri(names: tuple) -> str:
    """First existing file in `names` as a data-URI. Drop your own assets/bg.jpg to replace the skyline."""
    mimes = {".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg",
             ".jpeg": "image/jpeg", ".webp": "image/webp"}
    for n in names:
        p = ASSETS / n
        if p.exists():
            return f"data:{mimes[p.suffix.lower()]};base64," + base64.b64encode(p.read_bytes()).decode()
    return ""


def count(value: float, decimals: int = 0) -> str:
    """Number that counts up from 0 on load (pure CSS). The decimal part is static."""
    v = round(value, decimals)
    whole = int(v)
    frac = f"{v - whole:.{decimals}f}"[1:] if decimals else ""
    return f'<span class="count" style="--t:{whole}"></span>{frac}'


# --------------------------------------------------------------------------- model (mirrors the notebook)
@st.cache_resource(show_spinner="Training the model on the dataset...")
def load_artifacts():
    raw = pd.read_csv(DATA_PATH)
    df = raw.copy()

    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    df = df.drop("Applicant_ID", axis=1)
    clean = df.copy()  # readable version (Yes/No, text categories) for the insight charts

    le_edu, le_y = LabelEncoder(), LabelEncoder()
    df["Education_Level"] = le_edu.fit_transform(df["Education_Level"])
    df["Loan_Approved"] = le_y.fit_transform(df["Loan_Approved"])

    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    enc = pd.DataFrame(ohe.fit_transform(df[CAT_COLS]),
                       columns=ohe.get_feature_names_out(CAT_COLS), index=df.index)
    df = pd.concat([df.drop(columns=CAT_COLS), enc], axis=1)

    corr = df.select_dtypes(include="number").corr()["Loan_Approved"].drop("Loan_Approved")

    df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2
    X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"])
    y = df["Loan_Approved"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {"Naive Bayes": GaussianNB(),
              "Logistic Regression": LogisticRegression(),
              "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5)}
    metrics = {}
    for name, m in models.items():
        m.fit(X_train_s, y_train)
        pred = m.predict(X_test_s)
        metrics[name] = dict(acc=accuracy_score(y_test, pred), prec=precision_score(y_test, pred),
                             rec=recall_score(y_test, pred), f1=f1_score(y_test, pred),
                             cm=confusion_matrix(y_test, pred))

    approved = clean[clean["Loan_Approved"] == "Yes"]
    ref = approved[["Credit_Score", "DTI_Ratio", "Savings", "Applicant_Income"]].median()

    bounds = {}
    for c in ["Age", "Applicant_Income", "Coapplicant_Income", "Savings", "Collateral_Value", "Loan_Amount"]:
        bounds[c] = (int(math.floor(raw[c].min() / 100) * 100), int(math.ceil(raw[c].max() / 100) * 100))
    bounds["Age"] = (int(raw["Age"].min()), int(raw["Age"].max()))
    bounds["Credit_Score"] = (int(raw["Credit_Score"].min()), int(raw["Credit_Score"].max()))

    options = {c: sorted(clean[c].unique()) for c in CAT_COLS + ["Education_Level"]}

    return dict(models=models, metrics=metrics, scaler=scaler, ohe=ohe, le_edu=le_edu,
                columns=list(X.columns), clean=clean, corr=corr, ref=ref, bounds=bounds,
                options=options, n_records=len(raw), n_test=len(y_test),
                missing_pct=float(raw.isna().mean().mean() * 100),
                approval_rate=float((clean["Loan_Approved"] == "Yes").mean() * 100))


def predict(inputs: dict, art: dict, model_name: str):
    row = pd.DataFrame([inputs])
    row["Education_Level"] = art["le_edu"].transform(row["Education_Level"])
    enc = pd.DataFrame(art["ohe"].transform(row[CAT_COLS]),
                       columns=art["ohe"].get_feature_names_out(CAT_COLS), index=row.index)
    row = pd.concat([row.drop(columns=CAT_COLS), enc], axis=1)
    row["DTI_Ratio_sq"] = row["DTI_Ratio"] ** 2
    row["Credit_Score_sq"] = row["Credit_Score"] ** 2
    X = row.drop(columns=["Credit_Score", "DTI_Ratio"])[art["columns"]]
    Xs = art["scaler"].transform(X)
    model = art["models"][model_name]
    return float(model.predict_proba(Xs)[0, 1]), int(model.predict(Xs)[0])


# --------------------------------------------------------------------------- form state + sample profiles
FIELDS = {  # dataset column -> widget key
    "Age": "f_age", "Gender": "f_gender", "Marital_Status": "f_marital",
    "Education_Level": "f_education", "Dependents": "f_dependents",
    "Employment_Status": "f_emp", "Employer_Category": "f_employer",
    "Applicant_Income": "f_income", "Coapplicant_Income": "f_coincome",
    "Credit_Score": "f_credit", "DTI_Ratio": "f_dti", "Savings": "f_savings",
    "Existing_Loans": "f_existing", "Collateral_Value": "f_collateral",
    "Loan_Amount": "f_amount", "Loan_Term": "f_term",
    "Loan_Purpose": "f_purpose", "Property_Area": "f_area",
}
PRESETS = {
    "default": dict(f_age=35, f_gender="Male", f_marital="Married", f_education="Graduate",
                    f_dependents=1, f_emp="Salaried", f_employer="Private", f_income=11000,
                    f_coincome=4500, f_credit=715, f_dti=0.28, f_savings=10500, f_existing=1,
                    f_collateral=24000, f_amount=20000, f_term=48, f_purpose="Home",
                    f_area="Urban", f_model=FINAL_MODEL),
    "strong": dict(f_age=38, f_gender="Male", f_marital="Married", f_education="Graduate",
                   f_dependents=1, f_emp="Salaried", f_employer="MNC", f_income=15000,
                   f_coincome=6000, f_credit=770, f_dti=0.18, f_savings=16000, f_existing=0,
                   f_collateral=38000, f_amount=14000, f_term=36, f_purpose="Home", f_area="Urban"),
    "risky": dict(f_age=27, f_gender="Male", f_marital="Married", f_education="Not Graduate",
                  f_dependents=3, f_emp="Contract", f_employer="Business", f_income=4000,
                  f_coincome=1500, f_credit=585, f_dti=0.55, f_savings=800, f_existing=3,
                  f_collateral=6000, f_amount=33000, f_term=72, f_purpose="Personal", f_area="Rural"),
}


def load_preset(name: str) -> None:
    st.session_state.update(PRESETS[name])
    st.session_state.pop("result", None)


for _k, _v in PRESETS["default"].items():
    st.session_state.setdefault(_k, _v)


# --------------------------------------------------------------------------- styling
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Manrope:wght@400;500;600;700;800&display=swap');
@property --num { syntax: '<integer>'; initial-value: 0; inherits: false; }
:root { --ink:#0f1b36; --text:#2c3a58; --muted:#68758f; --card:rgba(255,255,255,.84); --line:rgba(15,27,54,.09);
        --gold:#b8821f; --gold-line:rgba(184,130,31,.28); --gold-soft:#fbf1dc; --mint:#0f9d84; --mint-soft:#e5f6f1;
        --coral:#d6454b; --coral-soft:#fdecec; --shadow:0 1px 2px rgba(15,27,54,.05), 0 14px 34px rgba(15,27,54,.08); }
html, body, .stApp { background:#ffffff !important; }
.stApp { font-family:'Manrope',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif; color:var(--text); overflow-x:hidden; }
.stApp::before { content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background-image:url("__BG__"); background-position:center center; background-size:cover; background-repeat:no-repeat; }
.stApp::after { content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background:radial-gradient(800px 460px at 10% -4%, rgba(15,157,132,.08), transparent 60%),
             radial-gradient(760px 460px at 100% 4%, rgba(217,164,65,.14), transparent 60%),
             linear-gradient(180deg, rgba(255,255,255,.62) 0%, rgba(255,255,255,.42) 45%, rgba(255,255,255,.12) 100%);
  animation:aurora 22s ease-in-out infinite alternate; }
@keyframes aurora { from { transform:translate3d(0,0,0) scale(1); } to { transform:translate3d(-2%,1.5%,0) scale(1.06); } }
[data-testid="stAppViewContainer"], [data-testid="stMain"], section.main { position:relative; z-index:1; background:transparent !important; }
[data-testid="stHeader"] { background:transparent !important; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
.block-container { max-width:760px !important; padding:1.4rem 1rem calc(3rem + env(safe-area-inset-bottom)) !important; }
.stApp p, .stApp label, .stApp li, .stApp input, .stApp button { font-family:'Manrope',system-ui,sans-serif; }

/* ---------- hero (the one page-load sequence) ---------- */
.hero { position:relative; overflow:hidden; border-radius:28px; padding:30px 26px 24px; margin:4px 0 18px;
  background:linear-gradient(135deg, rgba(255,255,255,.94), rgba(252,246,232,.86)); border:1px solid var(--gold-line);
  backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); box-shadow:var(--shadow), 0 30px 60px rgba(184,130,31,.10); }
.hero::before { content:""; position:absolute; left:26px; right:26px; top:0; height:3px; border-radius:0 0 4px 4px;
  background:linear-gradient(90deg, transparent, #d9a441, transparent); }
.hero-art { position:absolute; right:-90px; top:-70px; width:340px; height:340px; pointer-events:none; opacity:.8;
  background:url("__HERO__") center / contain no-repeat; animation:spin 120s linear infinite; }
@keyframes spin { to { transform:rotate(360deg); } }
.hero > *:not(.hero-art) { position:relative; z-index:1; }
.brand { display:inline-flex; align-items:center; gap:10px; font-family:'Fraunces',Georgia,serif; font-weight:700; font-size:1.15rem; color:var(--gold); }
.hero-title { font-family:'Fraunces',Georgia,serif; font-weight:700; font-size:clamp(1.95rem,6.4vw,2.9rem); line-height:1.08; color:var(--ink); margin:14px 0 10px; max-width:15ch; }
.lead { max-width:46ch; color:var(--text); font-size:1.02rem; line-height:1.55; margin:0 0 20px; }
.stats { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }
.stat { background:rgba(255,255,255,.85); border:1px solid var(--line); border-radius:16px; padding:12px 12px 10px; box-shadow:0 1px 2px rgba(15,27,54,.04); }
.stat b { display:block; font-family:'Fraunces',Georgia,serif; font-size:1.75rem; line-height:1.1; font-weight:700; color:var(--ink); }
.stat span { font-size:.78rem; color:var(--muted); }
.reveal { animation:rise .8s cubic-bezier(.2,.8,.2,1) both; animation-delay:calc(var(--i,0) * .12s); }
@keyframes rise { from { opacity:0; transform:translateY(14px); } }

/* counting numbers */
.count { --num:var(--t); counter-reset:num var(--num); animation:cnt 1.7s cubic-bezier(.2,.8,.2,1) both; }
.count::after { content:counter(num); }
@keyframes cnt { from { --num:0; } }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap:6px; background:rgba(15,27,54,.05); border:1px solid var(--line); border-radius:16px; padding:5px; }
.stTabs [data-baseweb="tab"] { flex:1 1 0; justify-content:center; height:44px; border-radius:12px; padding:0 8px; white-space:nowrap; color:var(--muted); }
.stTabs [data-baseweb="tab"] p { color:inherit !important; font-weight:700; font-size:.94rem; margin:0; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { background:#ffffff; color:var(--ink); box-shadow:0 2px 10px rgba(15,27,54,.12), inset 0 -2px 0 #d9a441; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top:16px; }

/* ---------- form ---------- */
[data-testid="stForm"] { border:0 !important; padding:0 !important; background:transparent !important; }
[class*="st-key-card_"] { background:var(--card); border:1px solid var(--line); border-radius:22px; padding:18px 16px 8px; margin-bottom:14px;
  backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); box-shadow:var(--shadow); }
.sec { display:flex; align-items:center; gap:12px; margin:2px 0 14px; }
.sec-ic { width:40px; height:40px; flex:none; border-radius:13px; display:grid; place-items:center; font-size:1.15rem; background:var(--gold-soft); border:1px solid var(--gold-line); }
.sec-t { font-family:'Fraunces',Georgia,serif; font-weight:700; font-size:1.18rem; color:var(--ink); line-height:1.2; }
.sec-s { font-size:.82rem; color:var(--muted); }
.hint { color:var(--muted); font-size:.93rem; margin:2px 0 12px; line-height:1.5; }
.stApp [data-testid="stWidgetLabel"] p { color:var(--text) !important; font-weight:600; font-size:.9rem; }
.stApp [data-baseweb="select"] > div, .stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"] {
  background:#ffffff !important; border-color:rgba(15,27,54,.15) !important; border-radius:12px !important; min-height:46px; }
.stApp input { color:var(--ink) !important; font-size:16px !important; }
.stApp [data-baseweb="select"] > div:focus-within, .stApp [data-baseweb="input"]:focus-within { border-color:#d9a441 !important; box-shadow:0 0 0 3px rgba(217,164,65,.2); }
.stApp [data-testid="stSliderThumbValue"], .stApp [data-testid="stSliderTickBarMin"], .stApp [data-testid="stSliderTickBarMax"] { color:var(--gold) !important; }
[data-testid^="stBaseButton-primary"] { width:100%; min-height:54px; border:0 !important; border-radius:16px; color:#ffffff !important;
  background:linear-gradient(135deg,#223662,#0f1b36) !important; box-shadow:inset 0 -3px 0 #d9a441, 0 12px 28px rgba(15,27,54,.24);
  transition:transform .15s ease, box-shadow .15s ease; }
[data-testid^="stBaseButton-primary"] p { color:#ffffff !important; font-weight:800; font-size:1.05rem; }
[data-testid^="stBaseButton-primary"]:hover { transform:translateY(-1px); box-shadow:inset 0 -3px 0 #d9a441, 0 16px 34px rgba(15,27,54,.30); }
[data-testid^="stBaseButton-primary"]:active { transform:translateY(1px) scale(.99); }
[data-testid^="stBaseButton-secondary"] { width:100%; min-height:48px; border-radius:14px; background:#ffffff !important; border:1px solid var(--line) !important; box-shadow:0 1px 2px rgba(15,27,54,.05); }
[data-testid^="stBaseButton-secondary"] p { color:var(--ink) !important; font-weight:700; }
[data-testid^="stBaseButton-secondary"]:hover { border-color:#d9a441 !important; }
[data-testid="stExpander"] { border:1px solid var(--line) !important; border-radius:14px !important; background:rgba(255,255,255,.7); }

/* ---------- result ---------- */
.result { position:relative; margin-top:18px; padding:26px 20px 22px; border-radius:26px; text-align:center; border:1px solid;
  backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px); animation:rise .7s cubic-bezier(.2,.8,.2,1) both; }
.result.ok { background:linear-gradient(160deg, #e3f6f0, #ffffff 66%); border-color:rgba(15,157,132,.38); box-shadow:0 18px 50px rgba(15,157,132,.16), var(--shadow); }
.result.no { background:linear-gradient(160deg, #fdeaea, #ffffff 66%); border-color:rgba(214,69,75,.36); box-shadow:0 18px 50px rgba(214,69,75,.13), var(--shadow); }
.ring-box { position:relative; width:190px; height:190px; margin:0 auto 6px; }
.ring { width:100%; height:100%; display:block; }
.ring-track { fill:none; stroke:rgba(15,27,54,.08); stroke-width:10; }
.ring-bar { fill:none; stroke-width:10; stroke-linecap:round; stroke-dasharray:339.292; stroke-dashoffset:var(--off); animation:ring 1.7s cubic-bezier(.2,.8,.2,1) both; }
.ok .ring-bar { stroke:var(--mint); filter:drop-shadow(0 3px 6px rgba(15,157,132,.35)); }
.no .ring-bar { stroke:var(--coral); filter:drop-shadow(0 3px 6px rgba(214,69,75,.3)); }
@keyframes ring { from { stroke-dashoffset:339.292; } }
.ring-num { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; font-family:'Fraunces',Georgia,serif; font-size:2.9rem; font-weight:700; color:var(--ink); line-height:1; }
.ring-num small { font-family:'Manrope',sans-serif; font-size:.78rem; font-weight:600; color:var(--muted); margin-top:6px; }
.verdict { font-family:'Fraunces',Georgia,serif; font-size:1.65rem; font-weight:700; margin:8px 0 6px; color:var(--ink); }
.ok .verdict { color:#0b7f6b; } .no .verdict { color:#b6343a; }
.verdict-sub { color:var(--text); max-width:42ch; margin:0 auto; line-height:1.55; font-size:.97rem; }

/* ---------- panels, bars, lists ---------- */
.panel { background:var(--card); border:1px solid var(--line); border-radius:22px; padding:18px 16px; margin-top:14px; backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); box-shadow:var(--shadow); }
.panel-t { font-family:'Fraunces',Georgia,serif; font-weight:700; font-size:1.12rem; color:var(--ink); margin-bottom:4px; }
.muted { color:var(--muted); font-size:.88rem; margin:0 0 6px; line-height:1.5; }
.cmp { margin-top:16px; }
.cmp-head { display:flex; justify-content:space-between; align-items:center; gap:8px; font-weight:700; color:var(--ink); font-size:.95rem; margin-bottom:4px; }
.cmp-head small { color:var(--muted); font-weight:500; }
.pill { font-size:.72rem; font-weight:700; padding:3px 10px; border-radius:999px; white-space:nowrap; }
.pill.good { background:var(--mint-soft); color:#0b7f6b; border:1px solid rgba(15,157,132,.3); }
.pill.warn { background:var(--coral-soft); color:#b6343a; border:1px solid rgba(214,69,75,.3); }
.pill.brass { background:var(--gold-soft); color:#8a5f0f; border:1px solid var(--gold-line); }
.bar-row { display:grid; grid-template-columns:78px 1fr 64px; align-items:center; gap:10px; margin:6px 0; font-size:.82rem; color:var(--muted); }
.bar-row b { color:var(--ink); text-align:right; font-weight:700; }
.bar { height:9px; border-radius:99px; background:rgba(15,27,54,.07); overflow:hidden; }
.bar > i { display:block; height:100%; width:var(--w); border-radius:99px; animation:grow 1.3s cubic-bezier(.2,.8,.2,1) both; }
.bar.you > i { background:linear-gradient(90deg,#0f9d84,#4cc9b0); }
.bar.ref > i { background:linear-gradient(90deg,#b8821f,#e0ae4f); }
@keyframes grow { from { width:0; } }
.tips { list-style:none; margin:10px 0 0; padding:0; }
.tips li { padding:10px 12px; border-radius:12px; background:#faf7ef; border-left:3px solid #d9a441; margin-bottom:8px; color:var(--text); font-size:.92rem; line-height:1.5; }
.fine { color:var(--muted); font-size:.8rem; line-height:1.5; margin-top:14px; }
.mrow { padding:12px 0; border-top:1px solid var(--line); }
.mrow:first-of-type { border-top:0; }
.mname { display:flex; align-items:center; gap:8px; flex-wrap:wrap; font-weight:700; color:var(--ink); margin-bottom:4px; }
.cm { display:grid; grid-template-columns:78px 1fr 1fr; gap:8px; margin-top:10px; }
.cmh { font-size:.76rem; color:var(--muted); display:flex; align-items:center; justify-content:center; text-align:center; line-height:1.3; }
.cmc { border-radius:14px; padding:16px 8px; text-align:center; font-family:'Fraunces',Georgia,serif; font-size:1.75rem; font-weight:700; }
.cmc small { display:block; font-family:'Manrope',sans-serif; font-size:.72rem; font-weight:600; opacity:.9; margin-top:2px; }
.cmc.ok { background:var(--mint-soft); border:1px solid rgba(15,157,132,.3); color:#0b7f6b; }
.cmc.bad { background:var(--coral-soft); border:1px solid rgba(214,69,75,.28); color:#b6343a; }
.steps { list-style:none; counter-reset:step; margin:12px 0 0; padding:0; }
.steps li { counter-increment:step; position:relative; padding:2px 0 16px 46px; color:var(--text); line-height:1.55; font-size:.95rem; }
.steps li::before { content:counter(step); position:absolute; left:0; top:0; width:32px; height:32px; border-radius:50%; display:grid; place-items:center;
  font-family:'Fraunces',Georgia,serif; font-weight:700; color:#f6e2b0; background:linear-gradient(135deg,#223662,#0f1b36); box-shadow:0 4px 10px rgba(15,27,54,.2); }
.steps li::after { content:""; position:absolute; left:15px; top:36px; bottom:2px; width:2px; background:rgba(184,130,31,.3); }
.steps li:last-child::after { display:none; }
.steps b { color:var(--ink); }
.chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
.foot { text-align:center; color:var(--muted); font-size:.82rem; margin-top:26px; line-height:1.7; }
.foot a { color:var(--gold); text-decoration:none; border-bottom:1px solid var(--gold-line); margin:0 6px; }

/* ---------- phones ---------- */
@media (max-width:640px) {
  .block-container { padding:1rem .8rem calc(3rem + env(safe-area-inset-bottom)) !important; }
  .hero { padding:22px 18px 18px; border-radius:22px; }
  .hero-art { width:240px; height:240px; right:-80px; top:-60px; opacity:.55; }
  .stat { padding:10px 10px 8px; } .stat b { font-size:1.4rem; } .stat span { font-size:.72rem; }
  [class*="st-key-card_"] { padding:16px 14px 6px; border-radius:20px; }
  .ring-box { width:168px; height:168px; } .ring-num { font-size:2.5rem; }
  .bar-row { grid-template-columns:64px 1fr 58px; gap:8px; }
  .cm { grid-template-columns:64px 1fr 1fr; }
}
@media (prefers-reduced-motion:reduce) {
  .stApp::after, .hero-art, .reveal, .result, .bar > i, .ring-bar, .count { animation:none !important; }
  .stApp * { transition:none !important; }
}
"""


def inject_css() -> None:
    bg = asset_uri(("bg.jpg", "bg.jpeg", "bg.png", "bg.webp", "bg_city.svg"))
    hero = asset_uri(("hero.png", "hero.webp", "hero_dial.svg"))
    css = CSS.replace("__BG__", bg).replace("__HERO__", hero)
    css = "\n".join(ln.strip() for ln in css.splitlines() if ln.strip())
    st.markdown("<style>\n" + css + "\n</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- UI pieces
def hero(art: dict) -> None:
    nb = art["metrics"][FINAL_MODEL]
    html(f"""
    <section class="hero">
    <div class="hero-art"></div>
    <div class="brand reveal" style="--i:0">
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 2.5l8 3v6.2c0 4.7-3.2 8.3-8 9.8-4.8-1.5-8-5.1-8-9.8V5.5l8-3z" fill="#0f1b36"/><path d="M8.2 12.2l2.7 2.7 5-5.4" stroke="#e8bd62" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
    CreditWise
    </div>
    <div class="hero-title reveal" style="--i:1" role="heading" aria-level="1">Will this loan be approved?</div>
    <p class="lead reveal" style="--i:2">Enter an applicant's details and the model estimates the chance of approval in seconds. It learned from {art['n_records']:,} past applications.</p>
    <div class="stats reveal" style="--i:3">
    <div class="stat"><b>{count(art['n_records'])}</b><span>applications studied</span></div>
    <div class="stat"><b>{count(nb['acc'] * 100, 1)}%</b><span>accuracy on unseen data</span></div>
    <div class="stat"><b>{count(nb['prec'] * 100, 1)}%</b><span>precision</span></div>
    </div>
    </section>
    """)


def section(icon: str, title: str, sub: str) -> None:
    html(f'<div class="sec"><div class="sec-ic">{icon}</div><div><div class="sec-t">{title}</div><div class="sec-s">{sub}</div></div></div>')


COMPARE = [  # label, column, higher_is_better, formatter
    ("Credit score", "Credit_Score", True, lambda v: f"{v:.0f}"),
    ("Debt-to-income", "DTI_Ratio", False, lambda v: f"{v:.2f}"),
    ("Savings", "Savings", True, lambda v: f"{v:,.0f}"),
    ("Applicant income", "Applicant_Income", True, lambda v: f"{v:,.0f}"),
]


def render_result(res: dict, art: dict) -> None:
    p, approved = res["prob"], res["label"] == 1
    pct = round(p * 100)
    if pct <= 0:
        pct_html, shown = "&lt;1", "under 1%"
    elif pct >= 100:
        pct_html, shown = "&gt;99", "over 99%"
    else:
        pct_html, shown = count(pct), f"{pct}%"
    tone = "ok" if approved else "no"
    verdict = "Likely to be approved" if approved else "Likely to be declined"
    sub = f"The {res['model']} model puts this application's chance of approval at {shown}."
    if abs(p - 0.5) < 0.10:
        sub += " It is a close call, so small changes could tip the result."

    rows, good, weak = [], [], []
    for label, col, higher, fmt in COMPARE:
        you, ref = float(res["inputs"][col]), float(art["ref"][col])
        ahead = you >= ref if higher else you <= ref
        close = abs(you - ref) <= 0.03 * ref   # within 3% of the median approved applicant
        pill_cls, pill_txt = ("brass", "About the same") if close else (("good", "Stronger") if ahead else ("warn", "Weaker"))
        scale = max(you, ref, 1e-9) * 1.05
        hint = "" if higher else " <small>(lower is better)</small>"
        rows.append(f"""
        <div class="cmp">
        <div class="cmp-head"><span>{label}{hint}</span><span class="pill {pill_cls}">{pill_txt}</span></div>
        <div class="bar-row"><span>You</span><div class="bar you"><i style="--w:{you / scale * 100:.1f}%"></i></div><b>{fmt(you)}</b></div>
        <div class="bar-row"><span>Approved</span><div class="bar ref"><i style="--w:{ref / scale * 100:.1f}%"></i></div><b>{fmt(ref)}</b></div>
        </div>""")
        if close:
            continue
        (good if ahead else weak).append(
            f"{label} of {fmt(you)} is better than the typical approved applicant ({fmt(ref)})." if ahead
            else f"{label} is {fmt(you)}; approved applicants typically show {fmt(ref)}.")

    nb_note = " Naive Bayes often gives very confident percentages, so read the number as a rough guide." if res["model"] == "Naive Bayes" else ""
    if approved:
        tips_title, tips = "What is working in this profile", good[:3]
    else:
        tips_title, tips = "Where this profile falls short", weak[:3]
    tips_html = ""
    if tips:
        tips_html = f'<div class="panel-t" style="margin-top:20px">{tips_title}</div><ul class="tips">' + "".join(f"<li>{t}</li>" for t in tips) + "</ul>"

    html(f"""
    <div class="result {tone}" id="result">
    <div class="ring-box">
    <svg class="ring" viewBox="0 0 140 140" role="img" aria-label="Approval chance {shown}">
    <circle class="ring-track" cx="70" cy="70" r="54"/>
    <circle class="ring-bar" cx="70" cy="70" r="54" transform="rotate(-90 70 70)" style="--off:{RING_C * (1 - p):.2f}"/>
    </svg>
    <div class="ring-num"><div>{pct_html}%</div><small>chance of approval</small></div>
    </div>
    <div class="verdict">{verdict}</div>
    <p class="verdict-sub">{sub}</p>
    </div>
    <div class="panel">
    <div class="panel-t">How you compare</div>
    <p class="muted">Bars compare this application with the median approved applicant in the dataset.</p>
    {''.join(rows)}
    {tips_html}
    <p class="fine">This is an estimate from a model trained on a sample dataset. Real lenders weigh many more factors.{nb_note}</p>
    </div>
    """)


def style_fig(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=35, r=8, t=18, b=45), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Manrope, sans-serif", color="#2c3a58", size=12),
                      legend=dict(orientation="h", y=1.12, x=0), bargap=0.25)
    fig.update_xaxes(gridcolor="rgba(15,27,54,.08)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(15,27,54,.08)", zeroline=False)
    return fig


def chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False})


# --------------------------------------------------------------------------- page
inject_css()
art = load_artifacts()
hero(art)

tab_predict, tab_insights, tab_about = st.tabs(["Check eligibility", "Model insights", "About"])

# ---------------------------------------------------------------- tab 1: predict
with tab_predict:
    html('<p class="hint">Fill in the details below, or load a sample profile to see how it works.</p>')
    b1, b2 = st.columns(2)
    b1.button("Load a strong profile", key="btn_strong", on_click=load_preset, args=("strong",))
    b2.button("Load a risky profile", key="btn_risky", on_click=load_preset, args=("risky",))

    bd, opt = art["bounds"], art["options"]
    with st.form("loan_form"):
        with st.container(key="card_about"):
            section("👤", "About the applicant", "Personal details")
            st.slider("Age", bd["Age"][0], bd["Age"][1], key="f_age")
            a, b = st.columns(2)
            a.selectbox("Gender", opt["Gender"], key="f_gender")
            b.selectbox("Marital status", opt["Marital_Status"], key="f_marital")
            a, b = st.columns(2)
            a.selectbox("Education", opt["Education_Level"], key="f_education")
            b.slider("Dependents", 0, 3, key="f_dependents")

        with st.container(key="card_work"):
            section("💼", "Work and income", "How the applicant earns")
            a, b = st.columns(2)
            a.selectbox("Employment status", opt["Employment_Status"], key="f_emp")
            b.selectbox("Employer type", opt["Employer_Category"], key="f_employer")
            a, b = st.columns(2)
            a.number_input("Applicant income", bd["Applicant_Income"][0], bd["Applicant_Income"][1], step=500, key="f_income")
            b.number_input("Co-applicant income", bd["Coapplicant_Income"][0], bd["Coapplicant_Income"][1], step=500, key="f_coincome")

        with st.container(key="card_finance"):
            section("🏦", "Financial health", "Credit and existing commitments")
            st.slider("Credit score", bd["Credit_Score"][0], bd["Credit_Score"][1], key="f_credit")
            st.slider("Debt-to-income ratio", 0.10, 0.60, step=0.01, key="f_dti",
                      help="Share of income that already goes to debt payments. Lower is better.")
            a, b = st.columns(2)
            a.number_input("Savings", bd["Savings"][0], bd["Savings"][1], step=500, key="f_savings")
            b.number_input("Collateral value", bd["Collateral_Value"][0], bd["Collateral_Value"][1], step=1000, key="f_collateral")
            st.slider("Existing loans", 0, 4, key="f_existing")

        with st.container(key="card_loan"):
            section("📄", "The loan", "What is being requested")
            a, b = st.columns(2)
            a.number_input("Loan amount", bd["Loan_Amount"][0], bd["Loan_Amount"][1], step=1000, key="f_amount")
            b.selectbox("Loan purpose", opt["Loan_Purpose"], key="f_purpose")
            st.select_slider("Loan term", options=[12, 24, 36, 48, 60, 72, 84], key="f_term",
                             format_func=lambda m: f"{m} months")
            st.selectbox("Property area", opt["Property_Area"], key="f_area")

        with st.expander("Advanced: choose the model"):
            st.selectbox("Prediction model", list(art["models"]), key="f_model",
                         help="Naive Bayes is the model the notebook settled on.")
            html('<p class="fine">Input ranges match the data the model learned from, so results stay meaningful.</p>')

        submitted = st.form_submit_button("Check eligibility", type="primary")

    if submitted:
        inputs = {col: st.session_state[key] for col, key in FIELDS.items()}
        prob, label = predict(inputs, art, st.session_state["f_model"])
        st.session_state["result"] = dict(prob=prob, label=label, inputs=inputs, model=st.session_state["f_model"])
        if label == 1:
            st.balloons()

    if "result" in st.session_state:
        render_result(st.session_state["result"], art)

# ---------------------------------------------------------------- tab 2: insights
with tab_insights:
    mets = art["metrics"]
    rows = ""
    for name, m in mets.items():
        tag = '<span class="pill brass">Used in the notebook</span>' if name == FINAL_MODEL else ""
        rows += f"""
        <div class="mrow">
        <div class="mname">{name} {tag}</div>
        <div class="bar-row"><span>Accuracy</span><div class="bar you"><i style="--w:{m['acc'] * 100:.1f}%"></i></div><b>{m['acc'] * 100:.1f}%</b></div>
        <div class="bar-row"><span>Precision</span><div class="bar ref"><i style="--w:{m['prec'] * 100:.1f}%"></i></div><b>{m['prec'] * 100:.1f}%</b></div>
        <div class="bar-row"><span>Recall</span><div class="bar you"><i style="--w:{m['rec'] * 100:.1f}%"></i></div><b>{m['rec'] * 100:.1f}%</b></div>
        </div>"""
    html(f"""
    <div class="panel" style="margin-top:0">
    <div class="panel-t">How the three models compare</div>
    <p class="muted">Same data and the same 80/20 split, scored on the {art['n_test']} applications the models never saw.</p>
    {rows}
    </div>
    """)

    cm = mets[FINAL_MODEL]["cm"]
    html(f"""
    <div class="panel">
    <div class="panel-t">{FINAL_MODEL}: where it was right and wrong</div>
    <p class="muted">Each box counts test applications by what really happened and what the model predicted.</p>
    <div class="cm">
    <div></div><div class="cmh">Predicted declined</div><div class="cmh">Predicted approved</div>
    <div class="cmh">Actually declined</div><div class="cmc ok">{cm[0][0]}<small>correct</small></div><div class="cmc bad">{cm[0][1]}<small>wrongly approved</small></div>
    <div class="cmh">Actually approved</div><div class="cmc bad">{cm[1][0]}<small>wrongly declined</small></div><div class="cmc ok">{cm[1][1]}<small>correct</small></div>
    </div>
    </div>
    """)

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    clean = art["clean"]
    with st.container(key="card_c1"):
        html('<div class="panel-t">Approval rate by credit score</div><p class="muted">Share of applications approved in each credit score band.</p>')
        c1_slot = st.container()
    bands = pd.cut(clean["Credit_Score"], bins=[549, 599, 649, 699, 749, 799],
                   labels=["550-599", "600-649", "650-699", "700-749", "750-799"])
    rate = (clean["Loan_Approved"] == "Yes").groupby(bands, observed=True).mean() * 100
    fig = go.Figure(go.Bar(x=[str(i) for i in rate.index], y=rate.values,
                           marker=dict(color=rate.values, cmin=0, cmax=100,
                                       colorscale=[[0, "#d6454b"], [.5, "#d9a441"], [1, "#0f9d84"]]),
                           text=[f"{v:.0f}%" for v in rate.values], textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}: %{y:.0f}% approved<extra></extra>"))
    fig.update_yaxes(range=[0, 115], ticksuffix="%", showgrid=False)
    with c1_slot:
        chart(style_fig(fig, 280))

    with st.container(key="card_c2"):
        html('<div class="panel-t">Credit scores of approved and declined applicants</div><p class="muted">Approved applicants cluster at higher scores.</p>')
        c2_slot = st.container()
    fig = go.Figure()
    for lab, colr in [("Yes", "#0f9d84"), ("No", "#d6454b")]:
        fig.add_trace(go.Histogram(x=clean.loc[clean["Loan_Approved"] == lab, "Credit_Score"], name="Approved" if lab == "Yes" else "Declined",
                                   nbinsx=20, marker_color=colr, opacity=.7))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="Credit score")
    with c2_slot:
        chart(style_fig(fig, 290))

    with st.container(key="card_c3"):
        html('<div class="panel-t">What moves the decision</div><p class="muted">Correlation of each input with approval in the notebook. Green pushes towards approval, red pushes against it.</p>')
        c3_slot = st.container()
    corr = art["corr"]
    top = corr.reindex(corr.abs().sort_values(ascending=False).index)[:8].sort_values()
    fig = go.Figure(go.Bar(x=top.values, y=[n.replace("_", " ") for n in top.index], orientation="h",
                           marker_color=["#0f9d84" if v > 0 else "#d6454b" for v in top.values],
                           hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
    fig.update_xaxes(
    zeroline=True,
    zerolinecolor="rgba(15,27,54,.35)"
)

fig.update_yaxes(
    automargin=True,
    ticklabelstandoff=10
)

with c3_slot:
    chart(style_fig(fig, 380))

# ---------------------------------------------------------------- tab 3: about
with tab_about:
    html(f"""
    <div class="panel" style="margin-top:0">
    <div class="panel-t">How a prediction is made</div>
    <p class="muted">The same steps as the project notebook, run in order.</p>
    <ol class="steps">
    <li><b>Clean.</b> Gaps in number columns are filled with the column average, and gaps in category columns with the most common value.</li>
    <li><b>Encode.</b> Education becomes 0 or 1, and the other categories become separate yes/no columns.</li>
    <li><b>Engineer.</b> Credit score and debt-to-income ratio are replaced by their squares to capture non-linear effects.</li>
    <li><b>Scale.</b> Every input is standardised using only the training split.</li>
    <li><b>Predict.</b> A Gaussian Naive Bayes model trained on 80% of the data returns the probability of approval.</li>
    </ol>
    </div>
    <div class="panel">
    <div class="panel-t">The data</div>
    <p class="muted">{art['n_records']:,} loan applications with 18 details each. About {art['missing_pct']:.0f}% of the values were missing and were filled during cleaning. {art['approval_rate']:.0f}% of the applications in the cleaned data were approved.</p>
    </div>
    <div class="panel">
    <div class="panel-t">Built with</div>
    <div class="chips"><span class="pill brass">Python</span><span class="pill brass">pandas</span><span class="pill brass">scikit-learn</span><span class="pill brass">Streamlit</span><span class="pill brass">Plotly</span></div>
    </div>
    <div class="panel">
    <div class="panel-t">Please note</div>
    <p class="muted">CreditWise is a learning project. Its output is a statistical estimate from a sample dataset and is not financial advice or a real lending decision.</p>
    </div>
    """)

links = ""
if GITHUB_URL:
    links += f'<a href="{GITHUB_URL}" target="_blank" rel="noopener">GitHub</a>'
if LINKEDIN_URL:
    links += f'<a href="{LINKEDIN_URL}" target="_blank" rel="noopener">LinkedIn</a>'
html(f'<div class="foot">Built by {AUTHOR}<br>{links}</div>')

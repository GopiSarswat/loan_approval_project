# Loan Approval Prediction

Predicts whether a loan application should be **approved** or **rejected**
based on applicant information (income, education, credit history, property
area, etc.).

## Project structure (kept minimal on purpose)

```
loan_approval_project/
├── data/
│   └── loan_data.csv       ← put the real Kaggle dataset here (optional)
├── train_model.py          ← FILE 1: cleans data, trains & compares models, saves model.pkl
├── app.py                  ← FILE 2: FastAPI server that loads model.pkl and returns predictions
├── frontend/
│   └── index.html          ← simple form: user enters data, sees the prediction
├── requirements.txt
└── README.md
```

Only two Python files matter for the ML/API logic:
- **`train_model.py`** — everything about training happens here.
- **`app.py`** — everything about serving predictions happens here.

## Dataset

Use the public **"Loan Prediction Problem Dataset"** from Kaggle /
Analytics Vidhya. Download it and save it as `data/loan_data.csv` with
these columns:

```
Gender, Married, Dependents, Education, Self_Employed,
ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
Credit_History, Property_Area, Loan_Status
```

> If `data/loan_data.csv` is missing, `train_model.py` automatically
> generates a small synthetic dataset with the same columns so you can
> run the whole pipeline immediately, then swap in the real data later.

## Setup

```bash
pip install -r requirements.txt
```

## Step 1 — Train the model

```bash
python train_model.py
```

This script:
1. **Handles missing values** — categorical columns are filled with the
   most frequent value (mode); numeric columns (income, loan amount,
   term, credit history) are filled with the median.
2. **Encodes categorical data** — `Gender`, `Married`, `Dependents`,
   `Education`, `Self_Employed`, and `Property_Area` are label-encoded.
3. **Trains multiple models** — Logistic Regression, Decision Tree, and
   Random Forest.
4. **Compares performance** — accuracy, precision, recall, and F1 score
   for each model, printed as a summary table.
5. **Explains false approvals vs. false rejections** — counts and
   explains both error types (see below), then saves the best model
   (by F1 score) to `model.pkl`.

## Step 2 — Run the prediction API

```bash
uvicorn app:app --reload
```

- API docs: http://127.0.0.1:8000/docs
- Endpoint: `POST /predict`

Example request body:

```json
{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "0",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 0,
  "LoanAmount": 150,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Property_Area": "Urban"
}
```

Example response:

```json
{
  "approved": true,
  "decision": "Approved",
  "confidence": 0.966,
  "approval_probability": 0.966
}
```

## Step 3 — Use the frontend

Open `frontend/index.html` directly in a browser (double-click it) while
`app.py` is running. Fill in the form and click **Check Approval** — the
page calls the API and shows the decision along with the approval
probability.

## Risk analysis: false approvals vs. false rejections

| Error type | What it means | Real-world cost |
|---|---|---|
| **False Approval** (False Positive) | A loan that should be **rejected** gets predicted **approved** | Direct financial/credit risk to the lender — the riskiest error |
| **False Rejection** (False Negative) | A loan that should be **approved** gets predicted **rejected** | Lost business, and an unfair outcome for a credit-worthy applicant |

Because false approvals carry direct monetary loss, a production system
usually tunes the decision threshold to reduce them first, even if that
means accepting a few more false rejections. `train_model.py` prints
both counts for every model so you can see this trade-off directly.

## Deployment thinking

- `train_model.py` and `app.py` are deliberately separate: training is a
  one-off (or periodic) batch job, while `app.py` only needs to load the
  already-trained `model.pkl` and respond to requests — this is the same
  split used in real ML deployments.
- The FastAPI server is stateless and can be containerized (Docker) or
  deployed behind any ASGI-compatible host as-is.
- The frontend is plain HTML/JS with no build step, so it can be hosted
  anywhere (or opened as a local file) as long as it can reach the API.

## Learning outcomes covered

- **Classification workflow** — end-to-end: raw data → cleaning →
  encoding → training → evaluation → serving.
- **Categorical encoding** — income brackets, education, and property
  information encoded for model consumption.
- **Risk analysis** — false approvals vs. false rejections explained and
  measured per model.
- **Model comparison** — Logistic Regression vs. Decision Tree vs.
  Random Forest, compared on accuracy/precision/recall/F1.
- **Deployment thinking** — a minimal, realistic split between a
  training script, a prediction API, and a frontend.

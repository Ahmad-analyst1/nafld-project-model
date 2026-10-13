import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

st.set_page_config(page_title="NAFLD / Liver Disease Prediction", page_icon="🩺", layout="wide")

COLUMNS = [
    "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
    "Alkaline_Phosphotase", "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase", "Total_Protiens", "Albumin",
    "Albumin_and_Globulin_Ratio", "Result"
]
FEATURES = COLUMNS[:-1]

st.title("🩺 Liver Disease Prediction System")
st.write("A demonstration machine-learning app based on the Indian Liver Patient Dataset (ILPD).")

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload the ILPD CSV file", type=["csv"])
    st.caption("Expected columns: Age, Gender, bilirubin, enzyme/protein measurements, and Result. A headerless ILPD file is supported.")
    st.divider()
    st.header("Model settings")
    test_size = st.slider("Test set size (%)", min_value=20, max_value=40, value=30, step=5) / 100
    random_state = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)

@st.cache_data
def read_dataset(file_bytes):
    from io import BytesIO
    raw = pd.read_csv(BytesIO(file_bytes))
    # Handle standard ILPD CSVs without a header.
    if not set(COLUMNS).issubset(set(raw.columns)):
        raw = pd.read_csv(BytesIO(file_bytes), header=None, names=COLUMNS)
    raw.columns = [str(c).strip() for c in raw.columns]
    missing = [c for c in COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError("Could not identify expected ILPD columns. Missing: " + ", ".join(missing))
    raw = raw[COLUMNS].copy()
    raw = raw.drop_duplicates()
    for col in COLUMNS:
        if col != "Gender":
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["Gender"] = raw["Gender"].astype(str).str.strip().str.title()
    raw = raw.replace([np.inf, -np.inf], np.nan)
    raw = raw.dropna(subset=["Result"])
    raw = raw[raw["Gender"].isin(["Female", "Male", "F", "M"])]
    raw["Gender"] = raw["Gender"].replace({"F": "Female", "M": "Male"})
    if raw.empty:
        raise ValueError("No usable rows found. Check that the uploaded file is the ILPD dataset.")
    # ILPD labels are commonly 1 = liver disease and 2 = no liver disease.
    raw["Result"] = raw["Result"].astype(int)
    return raw

if uploaded is None:
    st.info("Upload your **Indian Liver Patient Dataset (ILPD)** CSV in the sidebar to start. The dataset is not bundled with this GitHub project.")
    st.markdown("**Expected ILPD columns** (headerless files are accepted): Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio, Result.")
    st.stop()

try:
    df = read_dataset(uploaded.getvalue())
except Exception as exc:
    st.error(f"Unable to load dataset: {exc}")
    st.stop()

# Encode sex consistently for this run.
gender_encoder = LabelEncoder()
df["Gender"] = gender_encoder.fit_transform(df["Gender"])
X = df[FEATURES].copy()
y = df["Result"].copy()

if y.nunique() < 2 or y.value_counts().min() < 2:
    st.error("The target column must contain at least two classes, with at least two rows in each class.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state), stratify=y
)

def build_models():
    # Imputation and scaling are fitted only on training folds via pipelines.
    return {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced")
        ),
        "Decision Tree": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(random_state=int(random_state), class_weight="balanced")
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=200, random_state=int(random_state), class_weight="balanced")
        ),
        "Support Vector Machine": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            SVC(probability=True, class_weight="balanced", random_state=int(random_state))
        )
    }

models = build_models()
trained_models, results = {}, {}
cv_splits = min(5, int(y.value_counts().min()))
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    scores = cross_val_score(model, X, y, scoring="accuracy", cv=cv_splits)
    trained_models[name] = model
    results[name] = {
        "accuracy": accuracy_score(y_test, pred) * 100,
        "cv_accuracy": scores.mean() * 100,
        "prediction": pred,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=sorted(y.unique()))
    }

with st.sidebar:
    selected_model = st.selectbox("Select machine-learning model", list(models.keys()))

st.header("📊 Model performance")
performance_df = pd.DataFrame([
    {"Model": name, "Test Accuracy (%)": round(results[name]["accuracy"], 2),
     "Cross-validation (%)": round(results[name]["cv_accuracy"], 2)}
    for name in models
])
st.dataframe(performance_df, use_container_width=True, hide_index=True)

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(performance_df["Model"], performance_df["Cross-validation (%)"])
ax.set_ylabel("Accuracy (%)")
ax.set_title("Model comparison (cross-validation)")
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.header("🔍 Patient prediction")
st.caption("Enter measurements in the units used by the source dataset.")
c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    gender = st.selectbox("Gender", ["Female", "Male"])
    total_bilirubin = st.number_input("Total bilirubin", min_value=0.0, value=1.0)
    direct_bilirubin = st.number_input("Direct bilirubin", min_value=0.0, value=0.3)
with c2:
    alkaline = st.number_input("Alkaline phosphotase", min_value=0.0, value=200.0)
    alt = st.number_input("Alamine aminotransferase (ALT)", min_value=0.0, value=30.0)
    ast = st.number_input("Aspartate aminotransferase (AST)", min_value=0.0, value=30.0)
with c3:
    proteins = st.number_input("Total proteins", min_value=0.0, value=6.5)
    albumin = st.number_input("Albumin", min_value=0.0, value=3.5)
    ag_ratio = st.number_input("Albumin and globulin ratio", min_value=0.0, value=1.0)

if st.button("🔮 Predict", type="primary", use_container_width=True):
    gender_value = gender_encoder.transform([gender])[0]
    patient = pd.DataFrame([[
        age, gender_value, total_bilirubin, direct_bilirubin, alkaline,
        alt, ast, proteins, albumin, ag_ratio
    ]], columns=FEATURES)
    model = trained_models[selected_model]
    prediction = model.predict(patient)[0]
    st.subheader("Prediction result")
    if prediction == 1:
        st.error("Model output: Result = 1 (commonly labelled liver disease in ILPD).")
    elif prediction == 2:
        st.success("Model output: Result = 2 (commonly labelled no liver disease in ILPD).")
    else:
        st.info(f"Model output class: {prediction}")
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(patient)[0]
        prob_df = pd.DataFrame({"Class": model.classes_, "Probability (%)": np.round(probabilities * 100, 2)})
        st.dataframe(prob_df, hide_index=True, use_container_width=True)

st.header("📈 Confusion matrix")
cm = results[selected_model]["confusion_matrix"]
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
ax.set_xlabel("Predicted class")
ax.set_ylabel("Actual class")
ax.set_title(selected_model)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.header("🗃️ Dataset overview")
m1, m2, m3 = st.columns(3)
m1.metric("Rows after cleaning", len(df))
m2.metric("Input features", len(FEATURES))
m3.metric("Target classes", y.nunique())
st.dataframe(df.head(20), use_container_width=True, hide_index=True)

st.warning(
    "Research/educational demonstration only. This model is not a medical device and must not be used "
    "to diagnose, rule out, or guide treatment for liver disease. Predictions may be inaccurate and "
    "dataset/model performance does not establish clinical validity."
)

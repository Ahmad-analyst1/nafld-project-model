# Liver Disease Prediction — Streamlit

A GitHub-ready Streamlit app It compares Logistic Regression, Decision Tree, Random Forest, and SVM models, then allows a user to enter patient measurements and view a model prediction.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository (for example, `nafld-streamlit-app`).
2. Upload `app.py`, `requirements.txt`, and this `README.md` to the repository root.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/), sign in with GitHub, and choose **Create app**.
4. Select your repository, branch, and `app.py` as the main file.
5. Deploy. Open the resulting app URL.
6. In the app sidebar, upload the ILPD CSV file to load the dataset.

## Dataset format

The app does not include the dataset. Upload the Indian Liver Patient Dataset (ILPD) CSV from the sidebar. It accepts a headerless 11-column ILPD CSV or a CSV with these column names:

`Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio, Result`

Gender values should be `Female`/`Male` (or `F`/`M`). Common ILPD target labels are `1` and `2`.

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Important notes
- This app is for educational/research demonstration only and is not a diagnostic tool.
- Model scores are dataset-dependent and are not evidence of clinical validity.

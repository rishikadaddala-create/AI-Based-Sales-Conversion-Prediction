# AI-Based Sales Conversion Probability Prediction

An academic machine learning project predicting sales lead conversion probability using **Logistic Regression** and **Random Forest** models, with model selection driven by Stratified Cross-Validation on training data, explainability via **SHAP/LIME**, and an interactive **Streamlit** web application designed for deployment to Streamlit Cloud.

## Project Structure

```
sales-conversion-prediction/
├── data/              # Raw and processed datasets
├── notebooks/         # Exploratory notebooks (EDA, Feature Engineering, Training)
├── src/               # Reusable production modules
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   └── explain.py
├── models/            # Serialized Scikit-learn Pipeline artifacts (.joblib)
├── reports/           # Visual figures and performance benchmarks
├── app/               # Streamlit web application
├── tests/             # Pytest automated test suite
├── requirements.txt   # Environment dependencies
└── README.md
```

## Setup & Installation

1. **Clone repository and create virtual environment**:
   ```bash
   git clone <repo-url>
   cd sales-conversion-prediction
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Streamlit Application**:
   ```bash
   streamlit run app/streamlit_app.py
   ```

4. **Run Test Suite**:
   ```bash
   pytest tests/
   ```

## Methodology Summary

- **Pipeline Integrity**: All feature scaling, encoding, and imputation are wrapped inside Scikit-learn `Pipeline` objects to eliminate data leakage.
- **Cross-Validated Selection**: Hyperparameter tuning and model selection are conducted strictly on `X_train` using 5-Fold Stratified Cross-Validation.
- **Untouched Test Set**: Final performance metrics are computed once on the locked model using `X_test`.
- **Explainability**: SHAP (SHapley Additive exPlanations) values provide local and global feature attribution.

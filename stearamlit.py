import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

df = pd.read_csv("credit_risk_dataset.csv")

df["person_emp_length"] = df["person_emp_length"].fillna(
    df["person_emp_length"].median()
)
df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())
df = df.drop_duplicates()

df = df[df["person_age"] <= 100]
df = df[df["person_emp_length"] <= df["person_age"]]

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

onehot_cols = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file",
]
ordinal_cols = ["loan_grade"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            onehot_cols,
        ),
        (
            "ordinal",
            OrdinalEncoder(
                categories=[["A", "B", "C", "D", "E", "F", "G"]]
            ),
            ordinal_cols,
        ),
    ],
    remainder="passthrough",
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

selector = SelectKBest(score_func=f_classif, k=15)
X_train_selected = selector.fit_transform(X_train_encoded, y_train)
X_test_selected = selector.transform(X_test_encoded)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

rf_model = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

y_pred = rf_model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))


joblib.dump(preprocessor, "preprocessor.joblib")
joblib.dump(selector, "selector.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(rf_model, "rf_model.joblib")

print(
    "\n✅ تم حفظ الملفات بنجاح! هتلاقي 4 ملفات بتنتهي بـ .joblib ظهرت عندك في الفولدر."
)



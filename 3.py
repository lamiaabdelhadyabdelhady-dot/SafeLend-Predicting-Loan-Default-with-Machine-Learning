import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("/content/Electric_Vehicle_Population_Data.csv")

df.head()
df.shape
df.info()
df.describe(include="all")
df.dtypes
df.isnull().sum()
df.duplicated().sum()
df["County"] = df["County"].fillna(df["County"].mode()[0])

df["City"] = df["City"].fillna(df["City"].mode()[0])

df["Electric Utility"] = df["Electric Utility"].fillna(df["Electric Utility"].mode()[0])

df["Vehicle Location"] = df["Vehicle Location"].fillna(df["Vehicle Location"].mode()[0])
df["Postal Code"] = df["Postal Code"].fillna(df["Postal Code"].median())

df["Legislative District"] = df["Legislative District"].fillna(df["Legislative District"].median())

df["2020 Census Tract"] = df["2020 Census Tract"].fillna(df["2020 Census Tract"].median())

df.isnull().sum()
df.select_dtypes(include="object").columns

df["Electric Vehicle Type"] = df["Electric Vehicle Type"].str.strip()

df["Clean Alternative Fuel Vehicle (CAFV) Eligibility"] = df["Clean Alternative Fuel Vehicle (CAFV) Eligibility"].str.strip()

df["Electric Vehicle Type"] = df["Electric Vehicle Type"].str.strip()

df["Electric Utility"] = df["Electric Utility"].str.strip()

df["Vehicle Location"] = df["Vehicle Location"].str.strip()

df.head()

df.nunique()

df.columns

df.describe()

df.nunique().sort_values(ascending=False)
df_unsup = df.copy()

df_unsup = df_unsup.drop([
    "Electric Vehicle Type",
    "VIN (1-10)",
    "DOL Vehicle ID",
    "Vehicle Location",
    "Postal Code",
    "Legislative District",
    "2020 Census Tract",
    "City",
    "County",
    "Electric Utility",
    "Model"
], axis=1)

df_unsup.head()

df_unsup.columns
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df_unsup["Make"] = le.fit_transform(df_unsup["Make"])

df_unsup["Clean Alternative Fuel Vehicle (CAFV) Eligibility"] = le.fit_transform(
    df_unsup["Clean Alternative Fuel Vehicle (CAFV) Eligibility"]
)

df_unsup.head()

df_unsup = df_unsup.sample(n=30000, random_state=42)
from sklearn.preprocessing import StandardScaler, LabelEncoder

scaler = StandardScaler()
le = LabelEncoder()

df_unsup['State'] = le.fit_transform(df_unsup['State'])

X_scaled = scaler.fit_transform(df_unsup)
from sklearn.decomposition import PCA

pca = PCA(n_components=0.90, random_state=42)

X_pca = pca.fit_transform(X_scaled)

print(X_pca.shape)
plt.figure(figsize=(8,5))

plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')

plt.xlabel("Number of Components")

plt.ylabel("Cumulative Explained Variance")

plt.grid()

plt.show()
from sklearn.cluster import KMeans

inertia = []

for k in range(1,11):

    model = KMeans(n_clusters=k, random_state=42)

    model.fit(X_pca)

    inertia.append(model.inertia_)
    plt.figure(figsize=(6, 4))

    plt.plot(range(1, 11), inertia, marker='o')

    plt.xlabel("Number of Clusters")

    plt.ylabel("Inertia")

    plt.title("Elbow Method")

    plt.show()
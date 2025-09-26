# Counting Number of Features
import pandas as pd
df = pd.read_csv("laptopData.csv")
df = df.dropna(thresh=df.shape[1]//2)
df = df.fillna(df.median(numeric_only=True))
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])
    df[col] = df[col].astype("category").cat.codes
X = df.drop("Price", axis=1)
print("Number of features:", X.shape[1])

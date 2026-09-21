import os
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")   # Path to the registered tourism.csv inside the data folder
df.drop(columns=["CustomerID"], inplace=True)          # Drop the customer identifier column which is not a predictive feature

# Categorical columns are intentionally left as raw strings.
# Training pipeline one-hot-encodes them, and the Streamlit app also sends them.


target = "ProdTaken"                                   # Setting the name of the column to predict (whether customer purchased the package), 1 if the customer purchased the package, else 0
X = df.drop(columns=[target])
y = df[target]

# Keeping the (imbalanced) purchase ratio consistent across splits by using stratify
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("ProdTaken distribution in train:")
print(ytrain.value_counts())

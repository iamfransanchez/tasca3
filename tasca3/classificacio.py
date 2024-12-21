import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

df = sns.load_dataset("penguins")
df.dropna(inplace=True)

X = df.drop(columns="species")
y = df["species"]

categorical_features = ["island", "sex"]
numerical_features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

y = y.astype("category").cat.codes

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vec = DictVectorizer(sparse=False)
X_train_cat = vec.fit_transform(X_train[categorical_features].to_dict(orient="records"))
X_test_cat = vec.transform(X_test[categorical_features].to_dict(orient="records"))

scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[numerical_features])
X_test_num = scaler.transform(X_test[numerical_features])

X_train_prepared = pd.concat([
    pd.DataFrame(X_train_num, index=X_train.index),
    pd.DataFrame(X_train_cat, index=X_train.index)
], axis=1)

X_test_prepared = pd.concat([
    pd.DataFrame(X_test_num, index=X_test.index),
    pd.DataFrame(X_test_cat, index=X_test.index)
], axis=1)

models = {
    "LR": LogisticRegression(C=100.0, random_state=42, solver="lbfgs"),
    "SVM": SVC(kernel="linear", C=1.0, random_state=42, probability=True),
    "DT": DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=3, p=2, metric="minkowski")
}

for model_name, model in models.items():
    model.fit(X_train_prepared, y_train)
    
    with open(f"models/{model_name.replace(' ', '_').lower()}.pck", "wb") as f:
        pickle.dump((scaler, vec, model), f)
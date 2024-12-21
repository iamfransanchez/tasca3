import pickle
import pandas as pd
import seaborn as sns 
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATHS = {
    "lr": "models/lr.pck",
    "svm": "models/svm.pck",
    "dt": "models/dt.pck",
    "knn": "models/knn.pck",
}

models = {}
for name, path in MODEL_PATHS.items():
    with open(path, "rb") as f:
        models[name] = pickle.load(f)

categorical_features = ["island", "sex"]
numerical_features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

df = sns.load_dataset("penguins").dropna() 
species_mapping = dict(enumerate(df["species"].astype("category").cat.categories))

@app.route("/<model_name>", methods=["POST"])
def predict(model_name):

    if model_name not in models:
        return jsonify({"error": f"Model '{model_name}' not found"}), 404

    scaler, vec, model = models[model_name]

    data = request.json

    missing_features = [
        feature
        for feature in categorical_features + numerical_features
        if feature not in data
    ]
    if missing_features:
        return jsonify({"error": f"Missing features: {', '.join(missing_features)}"}), 400

    cat_data = vec.transform([{feature: data[feature] for feature in categorical_features}])
    num_data = scaler.transform([[data[feature] for feature in numerical_features]])
    input_data = pd.concat(
        [
            pd.DataFrame(num_data),
            pd.DataFrame(cat_data),
        ],
        axis=1,
    )

    prediction = model.predict(input_data)
    prediction_probabilities = model.predict_proba(input_data)

    predicted_species = species_mapping[int(prediction[0])]
    predicted_probability = prediction_probabilities[0][int(prediction[0])] * 100  # Porcentaje


    return jsonify({
        "prediction": predicted_species,
        "probability": round(predicted_probability, 2)
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=4000)

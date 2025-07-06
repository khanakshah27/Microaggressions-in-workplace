from flask import Flask, render_template, request, jsonify
from mgagg import load_model, predict_text

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("micui.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    model = load_model("microagg_model.pkl")
    prediction = predict_text(model, text)

    return jsonify({"isMicroaggression": prediction})

if __name__ == "__main__":
    app.run(debug=True)


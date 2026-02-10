
from flask import Flask, render_template, request
import pickle
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)

# Load model and vectorizer
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)

    if prediction[0] == 1:
        result = "SPAM MESSAGE 🚫"
    else:
        result = "NOT SPAM MESSAGE ✅"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run()

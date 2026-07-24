from flask import Flask, render_template, request
import os
from predict import predict_image
from fish_data import fish_info

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No image selected"

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    fish, confidence = predict_image(filepath)

    return render_template(
        "result.html",
        prediction=fish,
        confidence=round(confidence,2),
        image=filepath,
        info=fish_info.get(fish)
    )


if __name__ == "__main__":
    app.run(debug=True)
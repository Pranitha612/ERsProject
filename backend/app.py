from flask import Flask, request, jsonify
from flask_cors import CORS
from model import analyze_image

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running"})


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "Empty file"}), 400

        result = analyze_image(file.read())

        return jsonify(result)

    except Exception as e:
        print("ERROR:", e)  # 🔥 shows error in terminal
        return jsonify({"error": "Internal Server Error"}), 500


if __name__ == "__main__":
    app.run(debug=True)
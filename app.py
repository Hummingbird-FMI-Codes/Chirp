from flask import Flask, request, abort, jsonify

from model.model import ImageCaptioningLLM

import requests
from pydantic import BaseModel, ValidationError

app = Flask(__name__)

llm = ImageCaptioningLLM()

class UploadImageDto(BaseModel):
    timestamp: str
    lat: float
    lng: float


@app.route('/images', methods=["POST"])
def hello_world():
    imagePath = request.get_json()["imagePath"]
    if not imagePath:
        abort(400)
    caption = llm.generate_caption_from_path(imagePath)
    return jsonify({"caption": caption})

@app.route('/image', methods=["POST"])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "Image is required"}), 400
    image = request.files['image']

    try:
        data = UploadImageDto(
            timestamp=request.form['timestamp'],
            lat=float(request.form['lat']),
            lng=float(request.form['lng']),
        )
    except (ValidationError, ValueError) as e:
        return jsonify({"error": "Invalid input", "details": str(e)}), 400
    
    response = requests.post("http://localhost:3000/animal-data", json={
        "timestamp": data.timestamp,
        "lat": data.lat,
        "lng": data.lng,
        "specimenName": "unknown"
    })
    
    if response.status_code == 201:
        return jsonify({"message": "File processed successfully", "external_response": response.json()}), 200
    else:
        return jsonify({"error": "Failed to process file", "external_response": response.text}), response.status_code


if __name__ == '__main__':
    app.run(debug=True)

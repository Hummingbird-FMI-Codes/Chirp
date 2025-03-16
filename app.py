from flask import Flask, request, abort, jsonify

from model.model import ImageCaptioningLLM

app = Flask(__name__)

llm = ImageCaptioningLLM()


@app.route('/images', methods=["POST"])
def hello_world():
    imagePath = request.get_json()["imagePath"]
    if not imagePath:
        abort(400)
    caption = llm.generate_caption_from_path(imagePath)
    return jsonify({"caption": caption})


if __name__ == '__main__':
    app.run()

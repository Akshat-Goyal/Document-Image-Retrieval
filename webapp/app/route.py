import base64
import io
from pathlib import Path

import cv2
import numpy as np
from flask import jsonify, render_template, request, Blueprint, send_from_directory
from PIL import Image

from image_retrieval.config import IMG_DIR
from image_retrieval.main import ImageRetriever

ir = ImageRetriever()
ir.load_hash_table(filename="hash_table.pickle")

routes = Blueprint("routes", __name__)
files = list(Path(IMG_DIR).glob("*.png"))


@routes.route("/")
def root():
    """
    Main page
    """
    return render_template("main.html")


@routes.route("/searchUpload", methods=["POST"])
def searchUpload():
    """
    Queries for nearest neighbors
    """
    f = request.files["file-0"]

    img = Image.open(io.BytesIO(f.stream.read()))
    img = np.array(img).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if img is None:
        print("Error, couldn't open file")
        return jsonify(images={})

    print(f"Received file. Dimen: {img.shape}")

    ret = ir.query(img)
    images = []
    votes = []

    for i in ret:
        images.append(files[i[1]].name)
        votes.append(str(i[0]))

    ret = {"images": images, "votes": votes}
    print(ret)

    return jsonify(ret)


@routes.route("/static/image/<path:filename>")
def static_images(filename):
    return send_from_directory(IMG_DIR, filename)

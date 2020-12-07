import os
import sys
from os.path import join, dirname, realpath
from app import app
from flask import render_template, request, jsonify
import numpy as np
import pickle
from PIL import Image
import io
import cv2
import numpy as np

sys.path.insert(
    0,
    realpath(os.path.join(dirname(realpath(__file__)), "../../")),
)

from image_retrieval.main import ImageRetriever

ir = ImageRetriever()
ir.load_hash_table(filename='../hash_table.pickle')

@app.route("/")
def root():
    """
    Main page
    """
    return render_template("main.html")

@app.route("/searchUpload", methods=["POST"])
def searchUpload():
    """
    Queries for nearest neighbors
    """
    print(request.files["file-0"])
    f = request.files["file-0"]

    img = Image.open(io.BytesIO(f.stream.read()))
    img = np.array(img).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret = ir.query(img)
    images = []
    votes = []
    for i in ret:
        images.append(i[1])

    if img is None:
        print("Error, couldn't open file")
        return jsonify(images={})

    print(f"Received file. Dimen: {img.shape}")

    ret = jsonify(
        images=list(WEBAPP_LABELS[np.array(indices)][0]),
        distances=distances[0].tolist(),
    )
    return ret

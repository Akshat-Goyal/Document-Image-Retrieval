import os
import sys
from os.path import join, dirname, realpath
from pathlib import Path
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
from image_retrieval.config import IMG_DIR

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
    files = list(Path(IMG_DIR).glob("*.png"))
    for i in ret:
        images.append(files[i[1]].name)
        votes.append(str(i[0]))

    if img is None:
        print("Error, couldn't open file")
        return jsonify(images={})

    print(f"Received file. Dimen: {img.shape}")

    ret = jsonify(
        images=list(images),
        votes=list(votes),
    )
    return ret

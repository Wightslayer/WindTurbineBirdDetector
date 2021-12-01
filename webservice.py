# Webservice based on https://www.youtube.com/watch?v=BUh76-xD5qU&t=2284s

import os

import argparse
import os
import sys
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from yolor.detect import detect
from flask import Flask
from flask import request
from flask import render_template


app = Flask(__name__)
UPLOAD_FOLDER = 'C:\\Users\\Wight\\PycharmProjects\\BirdDetector\\WindTurbineBirdDetector\\DLWebService\\static'

MODEL = None
MODEL_CFG = '../yolor/cfg/yolor_p6.cfg'
MODEL_WEIGHTS = '../yolor/trained_weigths/my_model.pt'
MODEL_IMG_SIZE = 1280
DEVICE = 'cuda'

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            # pred = Make prediction
            return render_template("index.html", prediction=1, image_loc=image_file.filename)
    return render_template("index.html", prediction=0, image_loc=None)


if __name__ == "__main__":
    # Load model
    # model = Darknet(MODEL_CFG, MODEL_IMG_SIZE).cuda()
    # model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE)['model'])
    # #model = attempt_load(weights, map_location=device)  # load FP32 model
    # #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    # model.to(DEVICE).eval()
    # model.half()  # to FP16
    detect()
    print('done')
    app.run(port=12000, debug=True)

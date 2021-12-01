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

from models.models import *
from detect import detect
from PIL import Image

from flask import Flask
from flask import request
from flask import render_template

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'predictions', 'input')
RESULT_FOLDER = os.path.join('static', 'predictions', 'output')

MODEL = None
MODEL_CFG = 'cfg/yolor_p6.cfg'
MODEL_WEIGHTS = 'trained_weights/my_model.pt'
MODEL_IMG_SIZE = 1280
DEVICE = 'cuda'
OPT = None

def make_square(im, fill_color=(0, 0, 0, 0)):
    # Thank you Stephen:
    # https://stackoverflow.com/questions/44231209/resize-rectangular-image-to-square-keeping-ratio-and-fill-background-with-black/44231784
    w, h = im.size
    max_size = max(w, h)
    scale_factor = MODEL_IMG_SIZE / max_size
    new_w, new_h = int(scale_factor * w), int(scale_factor * h)
    resized_img = im.resize((new_w, new_h))
    new_im = Image.new('RGB', (MODEL_IMG_SIZE, MODEL_IMG_SIZE), fill_color)
    new_im.paste(resized_img, (int((MODEL_IMG_SIZE - new_w) / 2), int((MODEL_IMG_SIZE - new_h) / 2)))

    return new_im


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            original_image_location = os.path.join(
                UPLOAD_FOLDER,
                'original_' + image_file.filename
            )

            pp_image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )

            image_file.save(original_image_location)

            # Preprocess image
            img = Image.open(original_image_location)
            pp_img = make_square(img)
            pp_img.save(pp_image_location)

            # Make prediction
            OPT.source = os.path.join(UPLOAD_FOLDER, image_file.filename)
            with torch.no_grad():
                detect(opt=OPT, model=model)

            # Show prediction and original
            return render_template("index.html",
                                   model_in_loc=image_file.filename,
                                   model_out_loc=image_file.filename)
    return render_template("index.html", prediction=0, image_loc=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolor_p6.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='static/predictions/input/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='static/predictions/output/', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='cfg/yolor_p6.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/BirdDetector.names', help='*.cfg path')
    opt = parser.parse_args()

    OPT = opt


    # Load model
    model = Darknet(MODEL_CFG, MODEL_IMG_SIZE).cuda()
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE)['model'])
    #model = attempt_load(weights, map_location=device)  # load FP32 model
    #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    model.to(DEVICE).eval()
    model.half()  # to FP16

    app.run(port=12000, debug=True)


from darkflow.net.build import TFNet
import cv2

import urllib.request

import numpy as np

import os

from flask import Flask
from flask import request, flash, redirect, url_for

from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/tmp/photos/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

options = {"model": "/data/yolov2-food100.cfg", "load": "/data/yolov2-food100.weights", "labels": "/data/food100.names", "threshold": 0.1}

tfnet = TFNet(options)

@app.route('/predict_remote/')
def predict_remote():
    url_path = request.args.get('url')
    with urllib.request.urlopen(url_path) as url:
        with open('/tmp/temp.jpg', 'wb') as f:
            f.write(url.read())

    imgcv = cv2.imread('/tmp/temp.jpg')
    result = tfnet.return_predict(imgcv)
    return str(result)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict/', methods=['GET', 'POST'])
def predict_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            imgcv = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            result = tfnet.return_predict(imgcv)
            return str(result)


# @app.route('/predict/')
# def predict_base64():
#     img_str = request.args.get('img')
#     np_arr = np.fromstring(img_str, np.uint8)
#     imgcv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     print(type(imgcv))
#     result = tfnet.return_predict(imgcv)
#     return str(result)
#     return 'hello ' + str(type(imgcv))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

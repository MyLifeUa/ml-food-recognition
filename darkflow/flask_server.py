
from darkflow.net.build import TFNet
import cv2
import numpy as np
import base64
import urllib.request
from flask import Flask, request
app = Flask(__name__)

options = {"model": "/data/yolov2-food100.cfg", "load": "/data/yolov2-food100.weights", "labels": "/data/food100.names", "threshold": 0.1}

tfnet = TFNet(options)

@app.route('/predict')
def predict():

    if "image_b64" in request.args:
        image_b64 = request.args.get("image_b64")
        nparr = np.fromstring(base64.b64decode(image_b64), np.uint8)
        imgcv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        result = tfnet.return_predict(imgcv)
        return str(result)

    elif "url" in request.args:
        url_path = request.args.get("url")
        with urllib.request.urlopen(url_path) as url:
            with open("/tmp/temp.jpg", "wb") as f:
                f.write(url.read())
                
        imgcv = cv2.imread("/tmp/temp.jpg")
        result = tfnet.return_predict(imgcv)
        return str(result)

    else:
        return "Error", 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

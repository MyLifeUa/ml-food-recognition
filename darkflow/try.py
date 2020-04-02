from darkflow.net.build import TFNet
import cv2

options = {"model": "/data/yolov2-food100.cfg", "load": "/data/yolov2-food100.weights", "labels": "/data/food100.names", "threshold": 0.1}

tfnet = TFNet(options)

imgcv = cv2.imread("images/pizza.jpg")
result = tfnet.return_predict(imgcv)
print(result)

imgcv = cv2.imread("images/sushi.jpg")
result = tfnet.return_predict(imgcv)
print(result)

imgcv = cv2.imread("images/test.jpg")
result = tfnet.return_predict(imgcv)
print(result)

imgcv = cv2.imread("images/burger.jpg")
result = tfnet.return_predict(imgcv)
print(result)

imgcv = cv2.imread("images/pasta.jpg")
result = tfnet.return_predict(imgcv)
print(result)

imgcv = cv2.imread("images/bacalhau.jpeg")
result = tfnet.return_predict(imgcv)
print(result)

imgcv = cv2.imread("images/arroz-de-pato.jpg")
result = tfnet.return_predict(imgcv)
print(result)

imgcv = cv2.imread("images/bitoque.jpg")
result = tfnet.return_predict(imgcv)
print(result)

imgcv = cv2.imread("images/lasanha.jpg")
result = tfnet.return_predict(imgcv)
print(result)

# import urllib.request

# with urllib.request.urlopen('https://prod-wolt-venue-images-cdn.wolt.com/5dd153c99b5fe61fc520c3b7/d3fb2ea1e7451fe43a2ee6ccff4b531f') as f:
#     print('alive')
#     imgcv = cv2.imread(f)
#     result = tfnet.return_predict(imgcv)
#     print(result)

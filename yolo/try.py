from darkflow.net.build import TFNet
import cv2

options = {"model": "/data/yolov2-food100.cfg", "load": "/data/yolov2-food100.weights", "labels": "/data/food100.names", "threshold": 0.1}

tfnet = TFNet(options)

imgcv = cv2.imread("pizza.jpg")
result = tfnet.return_predict(imgcv)
print(result)

imgcv = cv2.imread("sushi.jpg")
result = tfnet.return_predict(imgcv)
print(result)

imgcv = cv2.imread("test.jpg")
result = tfnet.return_predict(imgcv)
print(result)

imgcv = cv2.imread("burger.jpg")
result = tfnet.return_predict(imgcv)
print(result)

imgcv = cv2.imread("pasta.jpg")
result = tfnet.return_predict(imgcv)
print(result)

imgcv = cv2.imread("bacalhau.jpeg")
result = tfnet.return_predict(imgcv)
print(result)

imgcv = cv2.imread("arroz-de-pato.jpg")
result = tfnet.return_predict(imgcv)
print(result)

imgcv = cv2.imread("bitoque.jpg")
result = tfnet.return_predict(imgcv)
print(result)

imgcv = cv2.imread("lasanha.jpg")
result = tfnet.return_predict(imgcv)
print(result)

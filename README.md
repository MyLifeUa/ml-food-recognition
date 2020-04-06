
# Food Recognition Module

We will be using YOLO Darkflow for this.

## Instructions

Simply run:
```
docker-compose up
```

A test command will be run automatically to indentify images you pass through URL:
`http://0.0.0.0:5000/predict?url=https://media-cdn.tripadvisor.com/media/photo-s/17/ba/a6/31/burger.jpg`

or through base64 encoding:
`http://0.0.0.0:5000/predict?image_b64=<base_64_enconding>`

or you can pass a local file:
```bash
curl -X POST -F "file=@/path/to/file" http://localhost:5000/predict
```

## Links

https://www.analyticsvidhya.com/blog/2020/02/learn-image-classification-cnn-convolutional-neural-networks-3-datasets/

https://nanonets.com/blog/multi-label-classification-using-deep-learning/

https://www.kaggle.com/pouryaayria/convolutional-neural-networks-tutorial-tensorflow

https://www.kaggle.com/kanncaa1/deep-learning-tutorial-for-beginners

wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz

### Install nvidia docker support (not working)

[Instructions](https://askubuntu.com/questions/799184/how-can-i-install-cuda-on-ubuntu-16-04)

1. [Download CUDA](https://developer.nvidia.com/cuda-downloads)
2. `sudo sh FILE --override`

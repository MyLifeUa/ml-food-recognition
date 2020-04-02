FROM tensorflow/tensorflow:1.5.0

RUN apt-get update && apt-get install -y \
    python-pip \
    cython \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget

RUN pip install --upgrade pip
RUN pip install opencv-python

RUN cd "/" && \
    git clone https://github.com/thtrieu/darkflow.git &&\
    cd darkflow && \
    sed -i'' -- 's/self.offset = 16/self.offset = 20/g' /darkflow/darkflow/utils/loader.py && \
    pip install . 

COPY . /data/

WORKDIR /data

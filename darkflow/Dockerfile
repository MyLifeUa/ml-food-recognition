FROM tensorflow/tensorflow:1.5.0-py3

RUN apt-get update && apt-get install -y \
    python-pip \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget

RUN pip install --upgrade pip
RUN pip install opencv-python
RUN pip install cython

RUN cd "/" && \
    git clone https://github.com/thtrieu/darkflow.git &&\
    cd darkflow && \
    sed -i'' -- 's/self.offset = 16/self.offset = 20/g' /darkflow/darkflow/utils/loader.py && \
    pip install . 

COPY . /data/

WORKDIR /data

RUN pip install -r requirements.txt

RUN mkdir /tmp/photos

EXPOSE 5000

CMD python3 flask_server.py

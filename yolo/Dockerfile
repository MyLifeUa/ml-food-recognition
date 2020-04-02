FROM loretoparisi/darknet:latest

RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get install -y build-essential \
  libssl-dev \
  libffi-dev \
  python3-dev \
  python3-pip \
  libopencv-dev \
  python-opencv \
  python-pip

RUN apt install -y libcanberra-gtk-module libcanberra-gtk3-module

WORKDIR /data

RUN pip install --upgrade pip
RUN pip install Pillow 
RUN apt install -y python-tk

# COPY ./*.py ./

# RUN python food100_generate_bbox_file.py
# RUN python food100_split_for_yolo.py
# RUN python food100_tk_label_bbox.py

RUN pip3 install numpy

WORKDIR /root/darknet
RUN sed -i'' -- 's/OPENCV=0/OPENCV=1/g' Makefile
RUN sed -i'' -- 's/CUDNN=1/CUDNN=0/g' Makefile
RUN make

ENTRYPOINT []
CMD ["/bin/bash"]

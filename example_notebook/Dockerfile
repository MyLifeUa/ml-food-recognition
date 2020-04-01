FROM continuumio/anaconda3

RUN /opt/conda/bin/conda install jupyter -y
RUN mkdir /opt/notebooks

ENTRYPOINT jupyter notebook --notebook-dir=/opt/notebooks/ --ip='*' --port=8888 --allow-root

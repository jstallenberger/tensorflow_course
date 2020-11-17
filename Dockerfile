FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

LABEL maintainer="Jozsef Stallenberger"

RUN apt-get update

RUN pip3 install --upgrade pip && \
    pip3 install joblib scikit-learn sklearn pytz pandas seaborn

EXPOSE 8888 6006

VOLUME /project
WORKDIR /project

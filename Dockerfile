FROM tensorflow/tensorflow:latest-gpu

LABEL maintainer="Jozsef Stallenberger"

RUN apt-get update

RUN pip3 install --upgrade pip && \
    pip3 install numpy matplotlib pandas scikit-learn seaborn

EXPOSE 6006

VOLUME /project
WORKDIR /project

FROM nvidia/cuda:11.7.0-base-ubuntu22.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get install -y libgl1
RUN apt-get install -y cmake

RUN apt install -y tcl
RUN apt-get install -y g++ && \
    apt-get install -y build-essential

# Update Python
RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt install -y python3.7 python3.7-distutils

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2

RUN update-alternatives --config python3


RUN apt install -y python3-pip



RUN python3.7 -m pip install --upgrade pip

WORKDIR /srv
ADD ./requirements.txt /srv/requirements.txt



RUN pip3 install -r requirements.txt

ADD . /srv/



RUN apt install -y libpython3.7-dev


WORKDIR /app
COPY . /app

CMD python3.7 objectDetectionYolo.py

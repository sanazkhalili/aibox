FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
ENV DEBIAN_FRONTEND noninteractive
FROM ubuntu:14.04.4
RUN apt-get update && apt-get install -y apt-transport-https
RUN echo 'deb http://private-repo-1.hortonworks.com/HDP/ubuntu14/2.x/updates/2.4.2.0 HDP main' >> /etc/apt/sources.list.d/HDP.list
RUN echo 'deb http://private-repo-1.hortonworks.com/HDP-UTILS-1.1.0.20/repos/ubuntu14 HDP-UTILS main'  >> /etc/apt/sources.list.d/HDP.list
RUN echo 'deb [arch=amd64] https://apt-mo.trafficmanager.net/repos/azurecore/ trusty main' >> /etc/apt/sources.list.d/azure-public-trusty.list


RUN apt-get install -y libgl1
RUN apt-get install -y cmake

RUN apt install -y tcl
RUN apt-get install -y g++ && \
    apt-get install -y build-essential

# Update Python
RUN apt-get install -y python3.7




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


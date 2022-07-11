FROM python:3.7

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


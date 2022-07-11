# syntax=docker/dockerfile:1
FROM python:3.7-alpine
WORKDIR /srv
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
RUN apk add --no-cache gcc musl-dev linux-headers

ADD ./requirements.txt /srv/requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5000
COPY . .
CMD ["flask", "run"]
RUN apt install -y python3-pip


ADD . /srv/

RUN apt install -y libpython3.7-dev


WORKDIR /app
COPY . /app

CMD python3.7 objectDetectionYolo.py


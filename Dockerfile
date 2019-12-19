FROM python:3.7

ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /app
ADD . /app

CMD bash run.sh

VOLUME /output

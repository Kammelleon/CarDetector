FROM python:3.9.13-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -r requirements.txt

COPY . .

CMD ["python3","manage.py","runserver","0.0.0.0:8000", "--insecure"]
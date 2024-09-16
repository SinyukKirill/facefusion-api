FROM python:3.10

WORKDIR /facefusion

RUN apt-get update
RUN apt-get install curl -y
RUN apt-get install ffmpeg -y

COPY . .
RUN python install.py --torch cpu --onnxruntime default

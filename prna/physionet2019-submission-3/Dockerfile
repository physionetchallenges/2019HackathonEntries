FROM python:3.6.4-slim

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER jrubin01@gmail.com

RUN apt-get update && apt-get install -y libgomp1
RUN pip install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl

## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet2019
COPY ./ /physionet2019
WORKDIR /physionet2019

## Install your dependencies here using apt-get etc.

## Do not edit if you have a requirements.txt
RUN pip install -r requirements.txt


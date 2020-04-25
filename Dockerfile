FROM continuumio/anaconda3

LABEL org.opencontainers.image.authors=nvisal1@students.towson.edu
LABEL org.opencontainers.image.created=04/03/20
LABEL org.opencontainers.image.title="Topic Model Builder Dockerfile"
LABEL org.opencontainers.image.url=
LABEL org.opencontainers.image.source=

ENV USERNAME=model-builder
ENV USER_GROUP=users

WORKDIR /model-builder/app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .





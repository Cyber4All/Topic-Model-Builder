### Topic Model Builder

## Summary

The Topic Model Builder is responsible for training 3 text classification models 

These are ...
1. A Convolutional Neural Network (CNN)
2. A Recurrent Neural Network (RNN)
3. A Bidirectional Recurrent Neural Network

The models are trained using the CLARK topic data that stored in MongoDB and Elasticsearch

All three of the models are pickled and stored in cloud blob storage

## Why Won't This Run?
This project requires a large zip that can be downloaded at https://nlp.stanford.edu/projects/glove/

This file should be unzipped and placed in the project as ./machine_learning/glove.6B

These files are too large for git to handle (maybe this can be put into a git submodule in the future?)

## Where is this deployed?

The Topic Model Builder only runs in AWS CodeBuild when the CLARK Topic Service triggers it via HTTPS
AWS CodeBuild pulls the source code in this repository and follows the instructions that are located in 
`buildspec.yml`.

## Updating This Project

This project does not have a dedicated CI pipeline for building and pushing new image versions
to Dockerhub. Once the update is made in the source code, you will need to build and push the updated image
from your machine.

## Running This Project
This project requires specific dependencies (mainly specific versions of Keras and Tensorflow) to run. Because of this, I recommend running this from within a Docker container. The included Dockerfile contains instructions for building the exact environment needed to run this project. You will notice that the Dockerfile does not have a CMD. You can run the project by passing the CMD "python index.py through a `docker run` command or by running it from within the container itself.

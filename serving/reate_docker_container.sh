#!/usr/bin/env bash

# Build the docker image (time to go get yourself a coffee, maybe a meal as well, this will take a while.)
docker build -t izhangzhihao/tensorflow-serving:latest -f ./serving/Dockerfile .

# Run up the Docker container in terminal
docker run -ti izhangzhihao/tensorflow-serving:latest — port=8999 - file_system_poll_wait_seconds=36000 — model_base_path=/work/awesome_model_directory

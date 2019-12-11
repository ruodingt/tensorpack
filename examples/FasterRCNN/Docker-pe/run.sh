# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env bash

IMAGE_NAME=${1:-"pearlii-awe-pack:latest"}


DIR=$(dirname "${PWD}")
DIR=$(dirname "${DIR}")

echo "Running docker image ${IMAGE_NAME}"
echo "${DIR}/data:/data"

docker run -it --runtime=nvidia --restart=always -v "${DIR}/data":/data:Z -v "${DIR}/logs":/logs:Z --name=vodka_awspack -p 6010:5000 -p 6012:22 -p 6016:6006 -p 6018:8888 -d ${IMAGE_NAME}

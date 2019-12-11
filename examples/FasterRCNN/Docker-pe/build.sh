# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env bash

DOCKER_FOLDER=${1:-"."}
BRANCH_NAME=${2:-"master"}
IMAGE_NAME=${3:-"pearlii-awe-pack:latest"}


# The BRANCH_NAME refers to the git pull that happens inside of the Dockerfile
echo "Building docker image ${IMAGE_NAME} on branch ${BRANCH_NAME}"
echo ""

docker build ${DOCKER_FOLDER} -t ${IMAGE_NAME} --build-arg CACHEBUST=$(date +%s) --build-arg BRANCH_NAME=${BRANCH_NAME} --build-arg user=root --build-arg password=makefog

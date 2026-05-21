#!/bin/bash

docker_user="your_dockerhub_username"

read -p "Do you want to build the nightly version of the quant_master image? (default is stable) (yes/no): " answer;
answer=$(echo "$answer" | tr '[:upper:]' '[:lower:]')

if [ "$answer" = "yes" ]; then
    # Build the nightly version of the quant_master image
    docker build --build-arg IS_STABLE=no -t quant_master_image -f ./Dockerfile .
    image_tag="nightly"
else
    # Build the stable version of the quant_master image
    docker build -t quant_master_image -f ./Dockerfile .
    image_tag="stable"
fi

read -p "Is it uploaded to docker hub? (default is no) (yes/no): " answer;
answer=$(echo "$answer" | tr '[:upper:]' '[:lower:]')

if [ "$answer" = "yes" ]; then
    # Log in to Docker Hub
    # If you are a new docker hub user, please verify your email address before proceeding with this step.
    docker login
    # Tag the Docker image
    docker tag quant_master_image "$docker_user/quant_master_image:$image_tag"
    # Push the Docker image to Docker Hub
    docker push "$docker_user/quant_master_image:$image_tag"
else
    echo "Not uploaded to docker hub."
fi

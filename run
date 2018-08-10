#! /bin/bash
# Run the 3blue1blue_mnist.py inside a docker container with tensorflow
docker run -it -p 6006:6006 -v $(pwd):/current gcr.io/tensorflow/tensorflow:latest-devel bash -c "python /current/3blue1blue_mnist.py"

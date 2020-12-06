# Deep Learning Tutorials 

Coding examples for the "Tensorflow 2.0: Deep Learning and Artificial Intelligence" Udemy course

## Coding environment usage

* `make image` creates a Tensorflow docker image with GPU support
* `make container` spins up a container named `tensorflow_env` with all GPU's enabled
* `docker exec -it tensorflow_env bash` brings up a shell inside the container
* `docker exec -it tensorflow_env python3 /project/01_example/example.py` runs the training within the container
* Don't forget to cleanup with `docker stop tensorflow_env` once you're done (it will stop and remove the container automatically)
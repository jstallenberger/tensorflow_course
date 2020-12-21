DOCKER_IMAGE = tensorflow-jokka
DOCKER_CONTAINER = tensorflow_env

.PHONY: image
image:
	docker build --rm -f Dockerfile -t $(DOCKER_IMAGE) .

.PHONY: container
container:
	docker run --gpus all \
	-it --rm -d \
	--name $(DOCKER_CONTAINER) \
	-p 6006:6006 \
	-v $(CURDIR):/project \
	$(DOCKER_IMAGE)

.PHONY: bash
bash:
	docker exec -it $(DOCKER_CONTAINER) bash

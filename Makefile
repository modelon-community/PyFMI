.PHONY: build build-dev-image test shell
DOCKER_IMAGE := pyfmi-dev
IN_DOCKER_IMG := $(shell test -f /.dockerenv && echo 1 || echo 0)

define _run
	@if [ $(IN_DOCKER_IMG) -eq 1 ]; then \
		$(1);\
	else \
		docker run \
		--rm $(2) \
		-v $(CURDIR):/src \
		${DOCKER_IMAGE} \
		$(1); \
	fi
endef

build-dev-image:
	docker build -t ${DOCKER_IMAGE} .

.venv:
	$(call _run, python3.11 -m venv .venv --system-site-packages)
	$(call _run, pip install pytest)

build: .venv
	$(call _run, python setup.py install --fmil-home=/usr)

test: build
	$(call _run, pytest)

shell:
	$(call _run, /bin/bash,-it)

.PHONY: build build-dev-image test shell
DOCKER_IMAGE := pyfmi-dev 

define _run
	docker run \
	--rm $(2) \
	-v $(CURDIR):/src \
	-w /src \
	${DOCKER_IMAGE} \
	$(1);
endef

build-dev-image:
	docker build -t ${DOCKER_IMAGE} .

.venv:
	$(call _run, python3.11 -m venv .venv --system-site-packages)
	$(call _run, pip install pytest)

build: .venv
	$(call _run, python setup.py install --fmil-home=/usr)

test: .venv
	$(call _run, pytest)

shell:
	$(call _run, ,-it)

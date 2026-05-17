.PHONY: build build-dev-image build-manylinux-image test shell compile-deps
DOCKER_IMAGE      := pyfmi-dev
MANYLINUX_IMAGE   := pyfmi-manylinux
IN_DOCKER_IMG     := $(shell test -f /.dockerenv && echo 1 || echo 0)

MESON_SETUP_ARGS := -Dfmil_prefix=/usr
PIP_SETUP_ARGS   := $(addprefix -Csetup-args=,$(MESON_SETUP_ARGS))

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

define _run_with_venv
	$(call _run, bash -c '. .venv/bin/activate && $(1)')
endef

build-dev-image:
	docker build -t ${DOCKER_IMAGE} .

build-manylinux-image:
	docker build -f Dockerfile.manylinux -t ${MANYLINUX_IMAGE} .

.venv: requirements.lock
	$(call _run, python3.11 -m venv .venv)
	$(call _run_with_venv, pip install -r requirements.lock)
	$(call _run, touch .venv)

build: .venv
	$(call _run_with_venv, pip install . -v $(PIP_SETUP_ARGS))

test: build
	$(call _run_with_venv, pytest tests/)

shell:
	$(call _run, /bin/bash,-it)

# Regenerate requirements.lock from pyproject.toml. Run after changing
# build-system requires or runtime dependencies; commit the resulting file.
compile-deps:
	$(call _run, python3.11 -m venv .venv)
	$(call _run_with_venv, pip install pip-tools)
	$(call _run_with_venv, pip-compile --extra=dev --output-file=requirements.lock pyproject.toml)

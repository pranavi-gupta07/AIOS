PY ?= .venv/bin/python
UV ?= uv

.PHONY: venv kernel test eval eval-quick daemon demo app clean

venv:
	$(UV) venv .venv
	$(UV) pip install --python $(PY) -e ".[eval,test]"

kernel:
	$(MAKE) -C kernel

test:
	$(PY) -m pytest -q

eval:
	$(PY) -m bench.run_eval

eval-quick:
	$(PY) -m bench.run_eval --quick

daemon:
	.venv/bin/aiosd --spawn-nodes

demo:
	$(PY) examples/demo_adaptation.py

app:
	$(PY) examples/app_aios.py

clean:
	$(MAKE) -C kernel clean
	rm -rf .pytest_cache build *.egg-info aios/__pycache__ bench/__pycache__ tests/__pycache__

venv:
	python3 -m venv .venv
	. .venv/bin/activate && python -m pip install -U pip setuptools wheel

install:
	. .venv/bin/activate && pip install -r requirements.txt && pip install -e .

test:
	. .venv/bin/activate && pytest -q

run:
	. .venv/bin/activate && python -m robo_trader.runner



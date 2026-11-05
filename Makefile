.PHONY: install lint test test-unit test-integration data-check prepare-fleurs train-quick train evaluate api ui docker publish-model box-setup box-check

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check src tests scripts app
	ruff format --check src tests scripts app

test:
	pytest -m "not gpu" --cov=vaakku --cov-report=term-missing

test-unit:
	pytest tests/unit -m "not gpu"

test-integration:
	pytest tests/integration -m "not gpu"

data-check:
	python scripts/prepare_data.py --config configs/data.yaml

prepare-fleurs:
	python scripts/prepare_data.py --config configs/data_fleurs.yaml

train-quick:
	python scripts/train.py --config configs/train_quick.yaml

train:
	python scripts/train.py --config configs/train.yaml

evaluate:
	python scripts/evaluate.py --config configs/evaluation.yaml

api:
	uvicorn vaakku.api.app:app --app-dir src --host 0.0.0.0 --port 8000

ui:
	streamlit run app/streamlit_app.py

docker:
	docker build -t vaakku:local .

publish-model:
	python scripts/publish_model.py --config configs/publish.yaml

box-setup:
	bash scripts/setup_aws_box.sh

box-check:
	bash scripts/verify_gpu.sh

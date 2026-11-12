.PHONY: install run-baseline run-pipeline evaluate submit clean

install:
	pip install -r requirements.txt
	pip install lightgbm catboost optuna scipy

run-baseline:
	python starter/baseline.py

run-pipeline:
	python pipeline.py

evaluate:
	cd evaluation && python run_evaluation.py

clean:
	rm -rf __pycache__ evaluation/reports/ *.pyc

python3 -m venv hw2_env
source hw2_env/bin/activate
poetry install
pre-commit install
pre-commit run -a
python3 mlops_tools/train.py
python3 mlops_tools/infer.py

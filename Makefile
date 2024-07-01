benchmark:
	python3 ao/benchmarks/matmul.py
	python3 ao/benchmarks/bmm.py
test:
	pytest
format:
	python3 -m black . --exclude 3rdparty
install:
	pip3 install -e .
.PHONY: tidy

tidy:
	black .
	isort .
	flake8 .

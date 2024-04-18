##
# Gammy
#
# @file
#

SHELL := /bin/bash

.PHONY: test
test:
	pytest -v
	python3 -m doctest -v README.md

.PHONY: pip-compile
pip-compile:
	pip-compile --resolver=backtracking \
				--verbose \
				--extra=test,dev \
				--output-file=requirements.txt ./setup.cfg

.PHONY: release
release:
	@python3 -c "import gammy; print(f'Current version: {gammy.__version__}')"
	@read -p "Enter new version (X.Y.Z): " version && \
	sed -i "s/version = .*/version = $$version/g" ./setup.cfg && \
	echo "__version__ = '$$version'" > ./gammy/__version__.py
	make test
	python3 -m build
	@read -s -p "PyPI secret: " password && echo && \
	python3 -m twine upload dist/* --repository-url https://upload.pypi.org/legacy/ --username __token__ --password $$password

# end

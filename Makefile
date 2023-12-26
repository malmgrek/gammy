##
# Gammy
#
# @file
#

.PHONY: test
test:
	pytest -v
	python3 -m doctest -v README.md


.PHONY: pip-compile
pip-compile:
	pip-compile --resolver=backtracking \
				--verbose \
				--extra=test,dev \
				--output-file=requirements.txt

.PHONY: release
release:
	@python3 -c "import gammy; print(f'Current version: {gammy.__version__}')"
	@read -p "Enter new version (X.Y.Z): " version && \
	sed -i "s/version = .*/version = $$version/g" ./setup.cfg && \
	echo "__version__ = '$$version'" > ./gammy/__version__.py
	make test

# end

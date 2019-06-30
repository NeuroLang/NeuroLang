PYTEST ?= pytest
CTAGS ?= ctags


flake8:
	@if command -v flake8 > /dev/null; then \
		echo "Running flake8"; \
		flake8 flake8 --ignore N802,N806 `find . -name \*.py | grep -v setup.py | grep -v /doc/`; \
	else \
		echo "flake8 not found, please install it!"; \
		exit 1; \
	fi;
	@echo "flake8 passed"

test:
	$(PYTEST) --cov=neurolang --cov-config=.coveragerc --cov-report=xml --cov-report=term-missing \
	  -vv neurolang neurolang --junitxml=utest.xml

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) --python-kinds=-i -R neurolang

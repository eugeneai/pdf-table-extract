.PHONY: develop setup run-tests tests test gdb-test pip

LPYTHON=python3
V=$(PWD)/../../$(LPYTHON)
VB=$(V)/bin
PYTHON=$(VB)/$(LPYTHON)
ROOT=$(PWD)
#INI=icc.linkgrammar
#LCAT=src/icc/linkgrammar/locale/

develop: pip setup

pip:
	pip install -r requirements.txt

setup:
	python setup.py develop

run-tests:
	nosetests -w src/icc/tests

tests:	run-tests

test:	setup run-tests

gdb-test: setup
	gdb --args $(PYTHON) $(VB)/nosetests -w src/icc/tests

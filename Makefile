.PHONY: build dist install uninstall update clean

SHELL := /bin/bash
PACKAGE := Ambitious
VERSION := $(shell \grep "version=" setup.py | sed -e "s/ *version='\(.*\)',/\1/")

build:
	python setup.py build

dist:
	python setup.py sdist

install: dist
	pip install dist/$(PACKAGE)-$(VERSION).tar.gz

uninstall:
	pip uninstall $(PACKAGE)

update: uninstall install

clean:
	rm -fr build dist $(PACKAGE).egg-info
	find . -type d -name "__pycache__" -print0 | xargs -0 -I {} rm -fr {}

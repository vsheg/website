all: update-submodules

update-submodules:
	git submodule update --recursive --remote --force
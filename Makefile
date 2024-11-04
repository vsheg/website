all: update-submodules

update-submodules:
	git submodule update --remote --merge
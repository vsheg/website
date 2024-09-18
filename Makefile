all: intermediate-ml

intermediate-ml-pull:
	git submodule update --remote --merge

intermediate-ml-index:
	cp intermediate-ml/README.md intermediate-ml/index.md

intermediate-ml: intermediate-ml-pull intermediate-ml-index
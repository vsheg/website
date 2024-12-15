all: update-submodules posts

update-submodules:
	git submodule update --recursive --remote --force

clean:
	rm -rf _site index_files*

posts:
	fd -p -g "**/posts/*/*.typ" --exec pandoc -o {.}.md {}
	fd -p -g "**/posts/*/*.md" --exec mv {} {.}.qmd


.PHONY: clean posts
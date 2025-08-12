SHELL := /bin/zsh

all: update-submodules posts uv

update-submodules:
	git submodule update --recursive --remote --force

clean:
	rm -rf _site || true
	rm -rf **/index_files/ || true
	rm -rf **/*.quarto_ipynb || true
	rm -rf site_libs* 404_files/ || true
	rm -rf **/index.html || true

posts:
	fd -p -g "**/posts/*/*.typ" --exec pandoc -o {.}.md {}
	fd -p -g "**/posts/*/*.md" --exec mv {} {.}.qmd

uv:
	uv sync

rss:
	mv _site/posts/index.xml _site/feed.xml

.PHONY: clean posts rss
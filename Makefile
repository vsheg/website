SHELL := /bin/zsh

all: update-submodules posts

update-submodules:
	git submodule update --recursive --remote --force

clean:
	rm -rf _site
	rm -rf **/index_files/
	rm -rf **/*.quarto_ipynb
	rm -rf site_libs* 404_files/

posts:
	fd -p -g "**/posts/*/*.typ" --exec pandoc -o {.}.md {}
	fd -p -g "**/posts/*/*.md" --exec mv {} {.}.qmd


rss:
	mv _site/posts/index.xml _site/feed.xml

.PHONY: clean posts rss
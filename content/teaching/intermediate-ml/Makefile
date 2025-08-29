TYP_FILES := $(shell find . -path "./*/*" -name "*.typ" | sort)
PDF_ROOT := https://vsheg.github.io/intermediate-ml

all: compile

compile:
	@for typ in $(TYP_FILES); do \
		dir=$$(dirname $$typ); \
		parent_dir=$$(basename $$dir); \
		typst compile $$typ --root .; \
		pdf_file=$${typ%.typ}.pdf; \
		if [ -f $$pdf_file ]; then \
			mv $$pdf_file $$dir/$$parent_dir.pdf; \
			echo "Compiled $$typ to $$dir/$$parent_dir.pdf"; \
		fi; \
		typst compile --format png --ppi 70 --pages 1 $$typ --root .; \
		png_file=$${typ%.typ}.png; \
		if [ -f $$png_file ]; then \
			mv $$png_file $$dir/_cover.png; \
			echo "Generated cover $$dir/_cover.png"; \
		fi; \
	done

readme:
	@echo "# ML materials" > README.md
	@echo "" >> README.md
	@echo "> This is a work-in-progress draft of intermediate-level machine learning materials." >> README.md
	@echo "Thanks to LLMs for the high quality; any errors are mine." >> README.md
	@echo "" >> README.md
	@echo '<table width="100%"><tr>' >> README.md
	@col=0; \
	for typ in $(TYP_FILES); do \
		dir=$$(dirname $$typ); \
		parent_dir=$$(basename $$dir); \
		pdf_url=$$(echo $$dir | sed 's|^\./|$(PDF_ROOT)/|')/$$parent_dir.pdf; \
		img_url=$$(echo $$dir | sed 's|^\./|$(PDF_ROOT)/|')/_cover.png; \
		echo "<td align=\"center\"><a href=\"$$pdf_url\"><img src=\"$$img_url\" width=\"120\"/></a><br/><a href=\"$$pdf_url\">$$parent_dir.pdf</a></td>" >> README.md; \
		col=$$((col+1)); \
		if [ $$col -eq 4 ]; then \
			echo "</tr><tr>" >> README.md; \
			col=0; \
		fi; \
	done; \
	echo "</tr></table>" >> README.md

clean:
	@for typ in $(TYP_FILES); do \
		dir=$$(dirname $$typ); \
		parent_dir=$$(basename $$dir); \
		pdf_file=$$dir/$$parent_dir.pdf; \
		png_file=$$dir/_cover.png; \
		if [ -f $$pdf_file ]; then \
			echo "Removing $$pdf_file"; \
			rm $$pdf_file; \
		fi; \
		if [ -f $$png_file ]; then \
			echo "Removing $$png_file"; \
			rm $$png_file; \
		fi; \
	done
PAPER = paper
PACKAGE = TRACK-FirstnameI.trackingno
SHELL = /bin/bash

export BIBINPUTS=.:../bib

## Main targets: paper, bib, clean, distclean
paper: $(PAPER).pdf

$(PAPER).pdf: FORCE
	pdflatex -shell-escape $(PAPER).tex
	bibtex -min-crossrefs=99 $(PAPER)
	pdflatex -shell-escape $(PAPER).tex
	pdflatex -shell-escape $(PAPER).tex

bib:
	bibtex -min-crossrefs=99 $(PAPER)

clean:
	rm -f figures/*.aux figures/*.log
	rm -f $(PAPER).{aux,bbl,blg,log,lpg,nav,out,snm,toc,sta}

distclean: clean
	rm -f figures/*.pdf
	rm -f $(PAPER).pdf

FORCE:

.PHONY: paper bib clean distclean FORCE


## Rules related to camera-ready copy submission package
## TODO: Adapt the script to the IJCAI instructions
## package: FORCE
## 	../scripts/aaai/make-aaai-package $(PAPER) $(BIBINPUTS) $(PACKAGE)

package-clean:
	rm -rf $(PAPER)-package
	rm -f $(PAPER)-package.{log,pdf,tgz}

clean: package-clean
.PHONY: package package-clean

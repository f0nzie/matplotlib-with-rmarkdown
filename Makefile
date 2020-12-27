# Makefile written by Alfonso R. Reyes
# Python Jupyter \
 PYTHON_ENV_DIR = \
 START_NOTEBOOK = \
 CHECKPOINTS =        # do we want to delete the Jupyter checkpoints folder? Specify folder \
 bookdown \
 BOOKDOWN_FILES = _bookdown_files \
 BOOKDOWN_FILES_DIRS = _bookdown_files    # remove folder(s) for clean rule \
 OUTPUT_DIR = .                           # parent folder for output \
 PUBLISH_BOOK_DIR = public                # remove folder for clean rule \
 FIGURE_DIR =         # do we have a folder for figure files? Specify folder \
 LIBRARY =            # do we have a temp folder with JS libraries? Specify folder \
 MAIN_RMD = rtorch-minimal-book.Rmd      # get rid off main.Rmd when cleaning 

SHELL := /bin/bash
# Python Jupyter
PYTHON_ENV_DIR = 
START_NOTEBOOK = 
CHECKPOINTS =
# bookdown
BOOKDOWN_FILES_DIRS = _bookdown_files
OUTPUT_DIR = .
PUBLISH_BOOK_DIR = public
FIGURE_DIR =
LIBRARY =
MAIN_RMD = matplotlib-with-rmarkdown.Rmd
# conda
CONDA_ENV_NAME = r-python
CONDA_TYPE = miniconda3
ENV_RECIPE = environment.yml
# Detect operating system. Sort of tricky for Windows because of MSYS, cygwin, MGWIN
ifeq ($(OS), Windows_NT)
    OSFLAG = WINDOWS
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S), Linux)
        OSFLAG = LINUX
    endif
    ifeq ($(UNAME_S), Darwin)
        OSFLAG = OSX
    endif
endif
# conda exists? Works in Linux
ifeq (,$(shell which conda))
    HAS_CONDA=False
else
    HAS_CONDA=True
	CONDA_BASE_DIR=$(shell conda info --base)
    MY_ENV_DIR=$(CONDA_BASE_DIR)/envs/$(CONDA_ENV_NAME)
    CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
endif
# conda environment exists?
ifeq (True,$(HAS_CONDA))
ifneq ("$(wildcard $(MY_ENV_DIR))","")
	HAS_ENVIRONMENT=True	
else
	HAS_ENVIRONMENT=False
endif
endif



# create conda environment
conda_create:
	source ${HOME}/${CONDA_TYPE}/etc/profile.d/conda.sh ;\
	conda deactivate
	conda remove --name ${CONDA_ENV_NAME} --all -y
	conda env create -f ${ENV_RECIPE}

conda_remove:
	source ${HOME}/${CONDA_TYPE}/etc/profile.d/conda.sh ;\
	conda deactivate
	conda remove --name ${CONDA_ENV_NAME} --all -y

# activate conda only if environment exists
conda_activate:
ifeq (True,$(HAS_CONDA))
ifneq ("$(wildcard $(MY_ENV_DIR))","") 
	source ${HOME}/${CONDA_TYPE}/etc/profile.d/conda.sh ;\
	conda activate $(CONDA_ENV_NAME)
else
	@echo ">>> Detected conda, but $(CONDA_ENV_NAME) is missing in $(CONDA_BASE_DIR). Install conda first ..."
endif
else
	@echo ">>> Install conda first."
	exit
endif

conda_deactivate:
	source ${HOME}/${CONDA_TYPE}/etc/profile.d/conda.sh ;\
	conda deactivate

bs4book_render:
	export RSTUDIO_PANDOC="/usr/lib/rstudio/bin/pandoc";\
	Rscript -e 'bookdown::render_book("index.Rmd", "bookdown::bs4_book")'

# use rstudio pandoc
# this rule sets the PANDOC environment variable from the shell
gitbook_render:
	export RSTUDIO_PANDOC="/usr/lib/rstudio/bin/pandoc";\
	Rscript -e 'bookdown::render_book("index.Rmd", "bookdown::gitbook")'

# use rstudio pandoc
# this rule sets the environment variable from R using multilines
build_book2:
	Rscript -e "\
	Sys.setenv(RSTUDIO_PANDOC='/usr/lib/rstudio/bin/pandoc');\
	bookdown::render_book('index.Rmd', 'bookdown::gitbook')"
	
open_book:
ifeq ($(OSFLAG), OSX)
    @open -a firefox  $(PUBLISH_BOOK_DIR)/index.html
endif
ifeq ($(OSFLAG), LINUX)
	@firefox  $(PUBLISH_BOOK_DIR)/index.html
endif
ifeq ($(OSFLAG), WINDOWS)
	@"C:\Program Files\Mozilla Firefox\firefox" $(PUBLISH_BOOK_DIR)/index.html
endif


# knit the book and then open it in the browser
.PHONY: bs4_book gitbook1 gitbook2
bs4_book: conda_activate bs4book_render open_book conda_deactivate
	
git_book: conda_activate gitbook_render open_book conda_deactivate

gitbook2: build_book2 open_book




git_push:
	git push
	git subtree push --prefix public origin gh-pages	



.PHONY: clean
clean: tidy
		find $(OUTPUT_DIR) -maxdepth 1 -name \*.tex -not -name 'preamble.tex' -delete
		find $(FIGURE_DIR) -maxdepth 1 -name \*.png -delete ;\
		$(RM) -rf $(BOOKDOWN_FILES_DIRS)
		if [ -f ${MAIN_RMD} ]; then rm -rf ${MAIN_RMD};fi ;\
		if [ -f ${LIBRARY} ]; then rm ${LIBRARY};fi ;\
		if [ -d ${PUBLISH_BOOK_DIR} ]; then rm -rf ${PUBLISH_BOOK_DIR};fi
		if [ -d ${CHECKPOINTS} ]; then rm -rf ${CHECKPOINTS};fi


# delete unwanted files and folders in bookdown folder
.PHONY: tidy
tidy:
		find $(OUTPUT_DIR) -maxdepth 1 -name \*.md -not -name 'README.md' -delete
		find $(OUTPUT_DIR) -maxdepth 1 -name \*-book.html -delete
		find $(OUTPUT_DIR) -maxdepth 1 -name \*.png -delete
		find $(OUTPUT_DIR) -maxdepth 1 -name \*.log -delete
		find $(OUTPUT_DIR) -maxdepth 1 -name \*.rds -delete
		find $(OUTPUT_DIR) -maxdepth 1 -name \*.ckpt -delete
		find $(OUTPUT_DIR) -maxdepth 1 -name \*.nb.html -delete	
		find $(OUTPUT_DIR) -maxdepth 1 -name _main.Rmd -delete


# provide some essential info about the tikz files
.PHONY: info
info:
	@echo "OS is:" $(OSFLAG)
	@echo "Bookdown publication folder:" $(PUBLISH_BOOK_DIR)
	@echo "Has Conda?:" ${HAS_CONDA}
	@echo "Conda environment:" ${CONDA_ENV_NAME}
	@echo "Conda Base  Dir:" ${CONDA_BASE_DIR}
	@echo "Environment Dir:" ${MY_ENV_DIR}
	@echo "Conda environment:" ${CONDA_ENV_NAME}
	@echo "Does Environment *${CONDA_ENV_NAME}* exist?" ${HAS_ENVIRONMENT}
	@echo "Active Conda environment:" ${CONDA_DEFAULT_ENV}
	
# Makefile for a basic latex project
.PHONY: all pdf pdf-clean clean
OBJ_DIR := ./obj
BIN_DIR := ./bin

# Build the project
all: pdf # TODO:  Add python build here

# Build the paper as a pdf
pdf:
	latexmk -pdf -synctex=true -aux-directory=$(OBJ_DIR) -output-directory=$(BIN_DIR) main.tex
# Remove any build artifacts generated from `make pdf`
pdf-clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Remove any build arifacts
clean: pdf-clean

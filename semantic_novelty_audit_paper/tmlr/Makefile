.PHONY: all clean reproduce-paper

PYTHON ?= python3
AUDIT_ROOT ?= $(abspath ../../external_semantic_overlap_audit)

all: main.pdf

main.pdf: main.tex appendix.tex references.bib \
		sections/clutrr_external_validation.tex \
		sections/clutrr_appendix.tex \
		generated/tmlr_strengthening_macros.tex \
		figures/fig_semantic_identity_framework.pdf \
		figures/fig_slr_semantic_overlap.pdf \
		figures/tmlrs_clutrr_heterogeneity.pdf \
		figures/tmlrs_internal_heterogeneity.pdf \
		figures/tmlrs_semantic_vs_surface.pdf \
		figures/tmlrs_slr_novelty_bounds.pdf \
		tmlr.sty tmlr.bst
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

figures/fig_semantic_identity_framework.pdf: scripts/make_semantic_identity_figure.py
	$(PYTHON) scripts/make_semantic_identity_figure.py

reproduce-paper:
	cd "$(AUDIT_ROOT)" && $(PYTHON) scripts/tmlr_strengthening/run_t2_clutrr_heterogeneity.py
	cd "$(AUDIT_ROOT)" && $(PYTHON) scripts/tmlr_strengthening/run_t3_internal_heterogeneity.py
	cd "$(AUDIT_ROOT)" && $(PYTHON) scripts/tmlr_strengthening/analyze_t6_semantic_surface.py
	cd "$(AUDIT_ROOT)" && $(PYTHON) scripts/tmlr_strengthening/run_t9_property_summary.py
	cd "$(AUDIT_ROOT)" && $(PYTHON) scripts/tmlr_strengthening/run_t10_novelty_bounds.py
	$(PYTHON) scripts/copy_strengthening_artifacts.py "$(AUDIT_ROOT)"
	$(PYTHON) scripts/make_semantic_identity_figure.py
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	latexmk -C

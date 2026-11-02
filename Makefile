PYTHON ?= python3
AUDIT := external_semantic_overlap_audit
PAPER := semantic_novelty_audit_paper/tmlr

.PHONY: reproduce-paper verify

reproduce-paper:
	$(PYTHON) $(AUDIT)/scripts/build_cross_benchmark_figures.py
	cp semantic_novelty_audit_paper/figures/*.pdf $(PAPER)/figures/
	$(PYTHON) $(PAPER)/scripts/make_semantic_identity_figure.py
	cd $(PAPER) && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	$(MAKE) verify

verify:
	cd $(AUDIT)/artifacts/clutrr_v3_v4_paper_ready && $(PYTHON) verify_bundle.py
	cd $(AUDIT) && $(PYTHON) -I scripts/semantic_audit_v1.py validate \
		analysis/quality-enhancement-v1/p4/tensor_logic.audit.json \
		analysis/quality-enhancement-v1/p4/occam_markov.audit.json

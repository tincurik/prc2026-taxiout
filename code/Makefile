# SPDX-License-Identifier: GPL-3.0-or-later
# One entry point for the pipeline: every target calls the stage scripts of pipeline/ in the order of README.md ("Stages"); nothing else is hidden here.
#   make help                               the targets
#   make data FILES=files_list.zip          stage 0: challenge archive, open external data, ADS-B ground points   (FILES = the organisers' archive or a directory of its parquet files)
#   make features                           stage 1: every label-free feature file
#   make members                            stage 2: the copy classifier and the 11 regressors (one CUDA GPU, about 5 GPU-hours)
#   make chain                              stage 3 on the 2025 side, then the five rehearsal phases (3b): each hides two 2025 months like the test months
#   make final                              stage 4 (test months), the bundle chain (3c / 3d), the four post-steps and the prediction file of one draw
#   make bag-draw DRAW=2                    stage 5a: one extra draw of the final chain in its own phase (finalb2; about 7 h on one GPU; DRAW=3 for the third)
#   make bag-file                           stage 5b: average the draws, then the plan-window step: the file that ships (K = BAG_K = 3 draws)
#   make final-bag                          stage 4 and 5: `make final`, then the draws 2 ... BAG_K one after the other, then `make bag-file`
#   make smoke [SMOKE_DAYS=3]               the whole pipeline on small synthetic data, CPU only, no network (8 days a month: about 45 minutes, about 12 GB of RAM, about 2 GB of disk)
#   make adsb DAYS=2026-01-05:2026-01-11    fetch the adsb.lol ground points of the days named (a range, a month YYYY-MM or a comma list); resumable and checksummed;
#                                           PLAN=1 prints the cost (GB read, disk, hours) and fetches nothing; VERIFY=1 re-hashes the files on disk
#   make test                               unit tests of the plan-window rules (twin window, duplicate pair, plan clip), synthetic rows, a few seconds, no data
# PY=<python> picks the interpreter (default python; it needs requirements.txt installed), THREADS the stackers' thread count (default 10).
# make -n <target> prints the commands without running them.
SHELL := /bin/bash
.SHELLFLAGS := -eo pipefail -c
.PHONY: help data features members chain final bag-draw bag-file final-bag smoke adsb test
export PY ?= python
export PYTHONDONTWRITEBYTECODE := 1   # no __pycache__ in the tree after a run
PHASES := pseudo2 pseudo3 pseudo4 pseudo5 pseudo6
SMOKE_DIR ?= $(or $(TMPDIR),/tmp)/prc_smoke
CHAIN_ID ?= ar_134naclh_tww
BAG_K ?= 3
SHIPPED := outputs/overnight_final/test/$(CHAIN_ID).parquet

help:
	@sed -n '2,/^# make -n/p' Makefile | sed 's/^# \{0,1\}//'

data:
	@test -n "$(FILES)" || { echo "usage: make data FILES=<files_list.zip or a directory of its parquet files>"; exit 2; }
	bash pipeline/00_data.sh "$(FILES)"

features:
	bash pipeline/10_features.sh

members:
	bash pipeline/20_members.sh

# stage 3 (2025 side), then stage 3b: each rehearsal phase refits every member on the other ten months and runs the chain on the two it hides
chain:
	CTX=0 bash pipeline/30_chain.sh
	for p in $(PHASES); do EC_PHASE=$$p bash pipeline/50_final_phase.sh data seed features members chain file; done

# stage 4 (the four test months), stage 3c / 3d (stream-transformer members in every phase, bundle chain with BND_NB=1), the four post-steps, the file check
final:
	EC_PHASE=final bash pipeline/50_final_phase.sh data seed features members chain
	BND_NB=1 bash pipeline/39_bnd1.sh base
	for p in $(PHASES) final; do BND_NB=1 EC_PHASE=$$p bash pipeline/39_bnd1.sh members; done
	for p in $(PHASES) final; do BND_NB=1 EC_PHASE=$$p bash pipeline/39_bnd1.sh chain; done
	bash pipeline/39_post_steps.sh pools
	bash pipeline/39_post_steps.sh run
	bash pipeline/40_final_file.sh $(SHIPPED)
	@echo "prediction file: $(SHIPPED:.parquet=)_wit_zf3.parquet"

# stage 5a: draw DRAW (2 ... BAG_K) of the seed bag: the same chain refitted at the seed offset 100 * DRAW, in its own phase; independent of the other draws, so two can run on two cards
bag-draw:
	@test -n "$(DRAW)" || { echo "usage: make bag-draw DRAW=2"; exit 2; }
	bash pipeline/45_bag_draw.sh "$(DRAW)"

# stage 5b: the K draws averaged row by row (bag_build.py), the plan-window step on the average, the file check; needs outputs/overnight_{final,finalb2..finalb<BAG_K>}/test/$(CHAIN_ID).parquet
bag-file:
	bash pipeline/46_bag_file.sh "$(CHAIN_ID)" "$(BAG_K)"
	@echo "shipped file: outputs/overnight_final/test/$(CHAIN_ID)_bag$(BAG_K)_wit_zf3.parquet"

final-bag: final
	for k in $$(seq 2 $(BAG_K)); do bash pipeline/45_bag_draw.sh $$k; done
	$(MAKE) bag-file

smoke:
	SMOKE_DAYS=$(or $(SMOKE_DAYS),8) bash smoke/run_smoke.sh "$(SMOKE_DIR)"

# unit tests of the plan-window rules (tests/test_artefact_rules.py)
test:
	$(PY) -m unittest discover -s tests -v

# the ground points of the days named in DAYS, with the cost stated first (scripts/adsb_days.py); the stage-0 script fetches the days of the challenge months the same way
adsb:
	@test -n "$(DAYS)" || { echo "usage: make adsb DAYS=<YYYY-MM-DD | YYYY-MM-DD:YYYY-MM-DD | YYYY-MM | comma list> [PLAN=1] [VERIFY=1] [ADSB_WORKERS=3]"; exit 2; }
	$(PY) scripts/adsb_days.py "$(DAYS)" --workers $(or $(ADSB_WORKERS),3) $(if $(PLAN),--plan) $(if $(VERIFY),--verify)

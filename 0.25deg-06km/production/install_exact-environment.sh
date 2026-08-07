#!/bin/bash
export PIP_CACHE_DIR=/scratch4/BMC/gsienkf/Bo.Huang/.cache/pip
export TMPDIR=/scratch4/BMC/gsienkf/Bo.Huang/tmp/pip-build

lconda
conda env create -f exact-environment.yaml
conda run -n eagle pip install --no-cache-dir --no-build-isolation flash-attn==2.7.4.post1

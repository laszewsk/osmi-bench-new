#!/bin/sh

export SR_LOG_LEVEL="QUIET"

export LANG=C.UTF-8
export LANGUAGE=
export LC_CTYPE="C.UTF-8"
export LC_NUMERIC="C.UTF-8"
export LC_TIME="C.UTF-8"
export LC_COLLATE="C.UTF-8"
export LC_MONETARY="C.UTF-8"
export LC_MESSAGES="C.UTF-8"
export LC_PAPER="C.UTF-8"
export LC_NAME="C.UTF-8"
export LC_ADDRESS="C.UTF-8"
export LC_TELEPHONE="C.UTF-8"
export LC_MEASUREMENT="C.UTF-8"
export LC_IDENTIFICATION="C.UTF-8"
export LC_ALL=


# python3 startdb.py
# lsof -i :6780
python3 benchmark.py
cat Inference-Benchmark/inferencer/*/*.out
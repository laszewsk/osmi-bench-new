#!/bin/sh

export SR_LOG_LEVEL="QUIET"

python3 startdb.py
lsof -i :6780
python3 benchmark.py
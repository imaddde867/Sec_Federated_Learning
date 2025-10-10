#!/bin/bash
conda create -n fl_privacy python=3.9
conda activate fl_privacy
pip install -r requirements.txt
python scripts/verify_setup.py
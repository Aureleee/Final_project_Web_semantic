#!/bin/bash
# 1. Create the venv
python -m venv venv_WDmS

# 2. Use the direct path to the venv's python to install everything
# This ensures it CANNOT install to the global/user folder
./venv_WDmS/Scripts/python.exe -m pip install --upgrade pip
./venv_WDmS/Scripts/python.exe -m pip install -r requirements.txt
./venv_WDmS/Scripts/python.exe -m spacy download en_core_web_trf
#!/bin/bash
# This script is part of the virusfinder scanning pipeline
## copy into a new directory and execute
## it will then set up the working environment in the current directory

mkdir Scanning
mkdir Scans_processed
mkdir Logs

for f in ~/AutoOMR_single_pages/Scripts/0*sh; do
    echo "copying $(basename $f)"
    cp $f .
done

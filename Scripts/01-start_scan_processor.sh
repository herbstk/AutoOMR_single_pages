#!/bin/bash
# This script is part of the virusfinder scanning pipeline
## this file should be executed after initialization of the working directory
## it starts the scan processor pipeline

CURD=$(pwd)
cd Scanning
gnome-terminal --title "scan processor (press Ctrl-C to STOP)" -- \
               bash -c "python3 ~/AutoOMR_single_pages/auto_omr_sp.py -v -d ~/AutoOMR_single_pages/Templates $CURD/Scans_processed 2>&1 | tee -a $CURD/Logs/01-start_scan_processor.log"

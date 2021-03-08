#!/bin/bash
# This script is part of the virusfinder scanning pipeline
## this script should be run once there batch of scans has been processed and scan images have been uploaded
## it will collect the automatically determined answers and transfer them to the database

gnome-terminal --title "collecting and uploading scan answers" -- \
               bash -c "~/AutoOMR_single_page/Scripts/collect_all_data.sh Scans_processed 2>&1 | tee -a Logs/03-upload_scan_results.log"

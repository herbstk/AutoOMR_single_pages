#!/bin/bash
# This script is part of the virusfinder scanning pipeline
## this script should is invoked by a runner script and usually shouldn't be run on its own
## it will collect and downsize page 1 and page 2 of scans and then upload to the virusfinder server

for d in "$1"/*; do
    if [[ -d "$d" ]]; then
        if [[ ! -f "$d"/collected_answers ]]; then
            echo "Collecting answers for directory $d..."
            Rscript ~/AutoOMR_single_page/Scripts/collect_sp_questionnaires.R $d
            touch "$d"/collected_answers
        else
            echo "Answers of directory $d already have been collected. Skipping..."
        fi
    fi
done
sleep 5





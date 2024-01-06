#!/bin/bash

# This script is used to clean the log files which are useless. 
# Whether a log file is useless is determined by whether there is a corresponding directory in figures_dir.
# Usage: ./clean_log.sh [log_dir] [figures_dir]
# If you don't specify the log_dir and figures_dir, the default values are "./log" and "./figures".
# Notice: Don't use this script when you are running the experiments.

log_dir="./log"
figures_dir=“./figures”

if [ $# -eq 2 ]; then
    log_dir=$1
    figures_dir=$2
fi

for file in "$log_dir"/*.log; do
    filename=$(basename "$file" | sed 's/\.[^.]*$//')

    if [ ! -d "$figures_dir/$filename" ]; then
        rm "$file"
    fi
done

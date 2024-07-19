# Copyright (c) 2023-2024 Yuxuan Shao

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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

find "$log_dir" -type f -name "*.log" | while read -r log_file; do
    sub_path=${log_file#"$log_dir"}
    filename=$(basename "$log_file" | sed 's/\.[^.]*$//')
    figure_path="$figures_dir$sub_path"

    if [ ! -d "$figure_path/$filename" ]; then
        rm "$log_file"
        dir_path=$(dirname "$log_file")
        while [ -z "$(ls -A "$dir_path")" ]; do
            rmdir "$dir_path"
            if [ "$dir_path" == "$log_dir" ]; then
                break
            fi
            dir_path=$(dirname "$dir_path")
        done
    fi
done

find "$figures_dir" -type d -empty -delete
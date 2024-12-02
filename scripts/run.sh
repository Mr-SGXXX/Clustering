# MIT License

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

# Check if the correct number of arguments is provided
if [ "$#" -ge 4 ]; then
    echo "Usage: $0 conda_env_name run_count [config_file] [device]"
    exit 1
fi

# Assign arguments to variables
conda_env_name=$1
run_count=$2

if [ "$#" -eq 3 ]; then
    config_file=$3
fi
else
    config_file="cfg/example.cfg"
fi

if [ "$#" -eq 4 ]; then
    device=$4
fi

# Activate the specified conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$conda_env_name"

# Navigate to the directory containing main.py
cd ..

# Check if the .temp directory exists, if not, create it
if [ ! -d ".temp" ]; then
    mkdir .temp
fi

# Get the current date and time in the format YYYYMMDD_HHMMSS
timestamp=$(date +"%Y%m%d_%H%M%S")
unique_name="config_${timestamp}.cfg"

# Copy the config file to the .temp directory with the unique name
cp "$config_file" ".temp/$unique_name"

# Update the config_file variable to point to the new copy
config_file=".temp/$unique_name"

# Run the Python script the specified number of times
for ((i=1; i<=run_count; i++))
do
    echo "Running iteration $i"
    if [ "$#" -eq 4 ]; then
        python main.py -cp "$config_file" -d "$device"
    fi
    else 
        python main.py -cp "$config_file"
    fi
done

# Remove the temporary config file
rm "$config_file"
# Deactivate the conda environment
conda deactivate
# Navigate to the directory containing main.py
cd scripts
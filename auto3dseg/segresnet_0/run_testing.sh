#!/bin/bash

# Define variables
DATA_ROOT_PATH="/RW/2024/monai_benchmark/test_data_sample"
JSON_OUTPUT_PATH="./test_auto3dseg_datalist.json"
CONFIG_FILE="./configs/hyper_parameters.yaml"

# Create JSON file using the Python function
python generate_json.py --data_root_path "$DATA_ROOT_PATH" --output_path "./work_dir/segresnet_0/$JSON_OUTPUT_PATH"

# Check if JSON file was created successfully
if [ ! -f "$JSON_OUTPUT_PATH" ]; then
  echo "Error: JSON file was not created."
  exit 1
fi

# Update hyper_parameters.yaml file
sed -i "s|data_file_base_dir: .*|data_file_base_dir: $DATA_ROOT_PATH|g" "$CONFIG_FILE"
sed -i "s|data_list_file_path: .*|data_list_file_path: $JSON_OUTPUT_PATH|g" "$CONFIG_FILE"

echo "Configuration updated successfully."
echo "RUN auto3dseg inference"

# Run inference script
python scripts/infer.py run --config_file="$CONFIG_FILE"

echo "Inference complete, do organ mask split and save to segmentations"

python create_organ_mask.py

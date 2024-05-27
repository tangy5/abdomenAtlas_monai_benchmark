import json
import os
import argparse

# def create_json(data_root_path, output_path):
#     # Example function to create a JSON file
#     data = {
#         "root": data_root_path,
#         "some_other_key": "some_value"
#     }

#     # Save JSON to output_path
#     with open(output_path, 'w') as json_file:
#         json.dump(data, json_file, indent=4)
        
def create_datalist_json(data_dir, output_json):
    datalist = {"testing": []}
    for case_dir in sorted(os.listdir(data_dir)):
        case_path = os.path.join(data_dir, case_dir)
        if os.path.isdir(case_path):
            image_path = os.path.join(case_path, "ct.nii.gz")
            if os.path.exists(image_path):
                datalist["testing"].append({"image": image_path})
#             label_path = os.path.join(case_path, "segmentations", "combined_all_labels.nii.gz")
#             if os.path.exists(image_path) and os.path.exists(label_path):
#                 datalist["validation"].append({"image": image_path, "label": label_path})
    
    with open(output_json, "w") as json_file:
        json.dump(datalist, json_file, indent=4)
    return output_json

        
def main():
    parser = argparse.ArgumentParser(description="Generate JSON file for MONAI benchmark.")
    parser.add_argument('--data_root_path', type=str, required=True, help="Root path of the data.")
    parser.add_argument('--output_path', type=str, required=True, help="Output path for the JSON file.")
    
    args = parser.parse_args()
    
    create_datalist_json(args.data_root_path, args.output_path)

if __name__ == "__main__":
    main()

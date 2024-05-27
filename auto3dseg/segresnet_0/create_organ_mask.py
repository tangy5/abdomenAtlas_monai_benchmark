import os
import numpy as np
import nibabel as nib

# Define organ mapping
organs = {
    1: "aorta",
    2: "gall_bladder",
    3: "kidney_left",
    4: "kidney_right",
    5: "liver",
    6: "pancreas",
    7: "postcava",
    8: "spleen",
    9: "stomach"
}

def process_masks(input_directory):
    # Walk through the input directory
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file == 'ct.nii.gz':
                subject_path = root
                mask_path = os.path.join(subject_path, file)
                
                mask_img = nib.load(mask_path)
                mask_data = mask_img.get_fdata()
                original_affine = mask_img.affine
                
                segmentations_dir = os.path.join(subject_path, "segmentations")
                os.makedirs(segmentations_dir, exist_ok=True)
                
                # Save each organ as a separate binary file
                for organ_id, organ_name in organs.items():
                    organ_mask = (mask_data == organ_id).astype(np.uint8)
                    organ_filename = os.path.join(segmentations_dir, f"{organ_name}.nii.gz")
                    nib.save(nib.Nifti1Image(organ_mask, original_affine), organ_filename)
                    print(f"Saved {organ_name} mask to {organ_filename}")

if __name__ == "__main__":
    input_directory = "./work_dir/segresnet_0/prediction_testing"
    process_masks(input_directory)

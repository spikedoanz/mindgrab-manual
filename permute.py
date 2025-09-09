import numpy as np
import itertools
import os
from save import save_array_to_nifti # Assuming save.py is in the same directory
from brainchop.niimath import conform # Assuming brainchop is installed

def generate_permutations(input_nifti_path, output_dir="permutations"):
    """
    Generates all 48 permutations of a 3D NIfTI volume and saves them.

    Args:
        input_nifti_path (str): The path to the input NIfTI file.
        output_dir (str): The directory where the permuted files will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Load the original volume
    volume, header = conform(input_nifti_path)
    print(f"Loaded input volume with shape: {volume.shape}")

    # The 3 axes are 0, 1, 2. There are 3! = 6 permutations.
    axis_permutations = list(itertools.permutations([0, 1, 2]))

    # Each of the 3 axes can be flipped or not. There are 2^3 = 8 combinations.
    flip_combinations = list(itertools.product([False, True], repeat=3))

    count = 0
    # Iterate through every combination of axis order and flips
    for axes in axis_permutations:
        for flips in flip_combinations:
            count += 1
            
            # 1. Apply axis permutation
            permuted_volume = np.transpose(volume, axes)
            
            # 2. Apply flips
            flip_slices = [slice(None)] * 3
            if flips[0]:
                flip_slices[0] = slice(None, None, -1)
            if flips[1]:
                flip_slices[1] = slice(None, None, -1)
            if flips[2]:
                flip_slices[2] = slice(None, None, -1)
            
            final_volume = permuted_volume[tuple(flip_slices)]

            # 3. Generate an interpretable filename
            axes_str = f"axes_{axes[0]}-{axes[1]}-{axes[2]}"
            
            flip_names = []
            if flips[0]: flip_names.append("x")
            if flips[1]: flip_names.append("y")
            if flips[2]: flip_names.append("z")
            
            flips_str = "flips_" + ("-".join(flip_names) if flip_names else "none")

            filename = f"perm_{count:02d}_{axes_str}_{flips_str}.nii.gz"
            output_path = os.path.join(output_dir, filename)

            # 4. Save the new NIfTI file
            # We use the original header information for saving.
            save_array_to_nifti(final_volume.astype(np.float32), output_path)
            print(f"Saved: {filename} with shape {final_volume.shape}")
            
    print(f"\nSuccessfully generated and saved all {count} permutations.")


# Example usage
if __name__ == "__main__":
    # Ensure you have a 'conformed.nii.gz' file in the same directory
    # or change the path to your input file.
    input_file = "conformed.nii.gz"

    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'")
        print("Please place your NIfTI file in the correct location or update the path.")
    else:
        print("\n" + "="*60)
        print("Starting permutation generation...")
        print("="*60)
        generate_permutations(input_file)
        print("\n" + "="*60)
        print("Permutation generation complete.")
        print("="*60)

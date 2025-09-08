import numpy as np
import nibabel as nib

def save_array_to_nifti(data, filename, voxel_size=(1.0, 1.0, 1.0)):
    """
    Save a numpy array to a NIfTI file with a default conformed affine matrix.
    
    Parameters:
    -----------
    data : numpy.ndarray
        The 3D or 4D numpy array to save
    filename : str
        Output filename (should end with .nii or .nii.gz)
    voxel_size : tuple, optional
        Voxel dimensions in mm (default: 1mm isotropic)
    """
    
    # Create a conformed affine matrix
    # This creates a standard RAS+ oriented affine with specified voxel sizes
    affine = np.array([
        [voxel_size[0], 0, 0, 0],
        [0, voxel_size[1], 0, 0], 
        [0, 0, voxel_size[2], 0],
        [0, 0, 0, 1]
    ])
    
    # Create NIfTI image object
    nifti_img = nib.Nifti1Image(data, affine)
    
    # Save to file
    nib.save(nifti_img, filename)
    print(f"Saved array with shape {data.shape} to {filename}")

# Example usage
if __name__ == "__main__":
    # Create a sample 3D array (e.g., 64x64x64 brain volume)
    sample_data = np.random.rand(64, 64, 64).astype(np.float32)
    
    # Save with default 1mm isotropic voxels
    save_array_to_nifti(sample_data, "output_volume.nii.gz")
    
    # Save with custom voxel size (e.g., 2mm x 2mm x 3mm)
    save_array_to_nifti(sample_data, "output_volume_custom.nii.gz", 
                       voxel_size=(2.0, 2.0, 3.0))
    
    # Example with 4D data (time series)
    timeseries_data = np.random.rand(64, 64, 32, 100).astype(np.float32)
    save_array_to_nifti(timeseries_data, "timeseries.nii.gz")

import nibabel as nib
import numpy as np
from tinygrad import Tensor
from save import save_array_to_nifti  # assuming you have this helper
from tiny_meshnet import load_meshnet      # from the code you shared earlier


def load_nifti_as_tensor(path: str) -> Tensor:
    """Load a NIfTI file and return as a Tinygrad Tensor with shape (C,D,H,W)."""
    nii = nib.load(path)
    volume = nii.get_fdata().astype(np.float32)
    
    # Reorder to (C, D, H, W) format. Assuming grayscale (1 channel).
    # Nibabel usually gives (H, W, D). We'll transpose to match model expectations.
    volume = np.transpose(volume, (2, 0, 1))  # (D, H, W)
    volume = np.expand_dims(volume, axis=0)   # add channel: (C=1, D, H, W)
    volume = np.expand_dims(volume, axis=0)   # add channel: (C=1, D, H, W)
    
    return Tensor(volume)


def main():
    config_file = "model.json"
    weight_file = "model.pth"
    input_file = "conformed.nii.gz"
    
    print("Loading model...")
    model = load_meshnet(
        config_fn=config_file,
        model_fn=weight_file,
        in_channels=1,
        channels=15,
        out_channels=2,
    )
    
    print("Loading input volume...")
    x = load_nifti_as_tensor(input_file)
    print(f"Input shape: {x.shape}")
    
    print("Running forward pass...")
    out = model(x, True)
    
    print(f"Output shape: {out.shape}")
    
    # Convert back to numpy for saving
    out_np = out.numpy()
    
    # If output has multiple channels, save each as its own NIfTI
    for c in range(out_np.shape[0]):
        save_array_to_nifti(out_np[c], f"output_c{c}.nii.gz")
        print(f"Saved output channel {c} to output_c{c}.nii.gz")


if __name__ == "__main__":
    main()

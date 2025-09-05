import numpy as np
from brainchop.niimath import conform
from pydawn import utils, webgpu

def apply(kernel_source, buffer_bytes):
    """
    Apply a WebGPU kernel to a buffer.
    
    Args:
        kernel_source: WGSL shader source code as string
        buffer_bytes: bytes buffer containing float32 data
    
    Returns:
        bytes: processed buffer as bytes
    """
    # Create adapter and device
    adapter = utils.request_adapter_sync(power_preference=webgpu.WGPUPowerPreference_HighPerformance)
    dev = utils.request_device_sync(adapter, [])
    
    original_size = len(buffer_bytes)
    num_elements = original_size // 4  # 4 bytes per float32
    
    # Align buffer size to 16-byte boundary (WebGPU requirement)
    aligned_size = ((original_size + 15) // 16) * 16
    
    # Pad buffer if necessary
    if aligned_size > original_size:
        padded_bytes = buffer_bytes + b'\x00' * (aligned_size - original_size)
    else:
        padded_bytes = buffer_bytes
    
    # Create shader module
    shader_module = utils.create_shader_module(dev, kernel_source)
    
    # Create input and output buffers
    input_buffer = utils.create_buffer(
        dev, 
        aligned_size, 
        webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopyDst
    )
    output_buffer = utils.create_buffer(
        dev, 
        aligned_size, 
        webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopySrc
    )
    
    # Write data to input buffer
    utils.write_buffer(dev, input_buffer, 0, bytearray(padded_bytes))
    
    # Setup bind group layout
    binding_layouts = [
        {
            "binding": 0,
            "visibility": webgpu.WGPUShaderStage_Compute,
            "buffer": {"type": webgpu.WGPUBufferBindingType_ReadOnlyStorage},
        },
        {
            "binding": 1,
            "visibility": webgpu.WGPUShaderStage_Compute,
            "buffer": {"type": webgpu.WGPUBufferBindingType_Storage},
        },
    ]
    
    # Setup bindings
    bindings = [
        {
            "binding": 0,
            "resource": {"buffer": input_buffer, "offset": 0, "size": aligned_size},
        },
        {
            "binding": 1,
            "resource": {"buffer": output_buffer, "offset": 0, "size": aligned_size},
        },
    ]
    
    # Create bind group and pipeline layout
    bind_group_layout = utils.create_bind_group_layout(device=dev, entries=binding_layouts)
    pipeline_layout = utils.create_pipeline_layout(device=dev, bind_group_layouts=[bind_group_layout])
    bind_group = utils.create_bind_group(device=dev, layout=bind_group_layout, entries=bindings)
    
    # Create compute pipeline
    compute_pipeline = utils.create_compute_pipeline(
        device=dev,
        layout=pipeline_layout,
        compute={"module": shader_module, "entry_point": "main"},
    )
    
    # Create and run command encoder
    command_encoder = utils.create_command_encoder(dev)
    compute_pass = utils.begin_compute_pass(command_encoder)
    utils.set_pipeline(compute_pass, compute_pipeline)
    utils.set_bind_group(compute_pass, bind_group)
    
    # FIXED: Proper workgroup dispatch handling WebGPU limits
    workgroup_size = 256
    total_workgroups = (num_elements + workgroup_size - 1) // workgroup_size
    
    # WebGPU has a limit of 65535 workgroups per dimension
    max_workgroups_1d = 65535
    
    if total_workgroups <= max_workgroups_1d:
        # Use 1D dispatch
        workgroups_x = total_workgroups
        workgroups_y = 1
        workgroups_z = 1
    else:
        # Use 2D dispatch to handle more workgroups
        workgroups_x = max_workgroups_1d
        workgroups_y = (total_workgroups + max_workgroups_1d - 1) // max_workgroups_1d
        workgroups_z = 1
    
    utils.dispatch_workgroups(compute_pass, workgroups_x, workgroups_y, workgroups_z)
    
    utils.end_compute_pass(compute_pass)
    cb_buffer = utils.command_encoder_finish(command_encoder)
    
    # Submit command buffer
    utils.submit(dev, [cb_buffer])
    
    # Wait for GPU work to complete using sync
    utils.sync(dev)
    
    # Read result and trim back to original size
    result_buffer = utils.read_buffer(dev, output_buffer)
    return result_buffer[:original_size]


if __name__ == "__main__":
    # Load and conform the NIfTI file
    volume, header = conform("conformed.nii.gz")
    tensor = volume.transpose((2, 1, 0)).astype(np.float32)
    print(f"Tensor shape: {tensor.shape}")
    
    # Convert tensor to bytes
    tensor_bytes = tensor.tobytes()
    
    # Your original identity kernel works perfectly now!
    kernel = f"""
        @group(0) @binding(0)
        var<storage,read> input_data: array<f32>;
        
        @group(0) @binding(1)
        var<storage,read_write> output_data: array<f32>;
        
        @compute
        @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let index = global_id.x;
            let max_elements = {tensor.size}u;
            
            if (index >= max_elements) {{
                return;
            }}
            
            output_data[index] = input_data[index];
        }}
    """
    
    # Apply kernel
    result_bytes = apply(kernel, tensor_bytes)
    
    # Convert back to numpy
    result_tensor = np.frombuffer(result_bytes, dtype=np.float32).reshape(tensor.shape)
    
    print(f"Result shape: {result_tensor.shape}")
    print(f"Identity check: {np.allclose(tensor, result_tensor)}")
    
    # Example of a more interesting kernel - element-wise operations
    print("\n--- Testing element-wise multiplication ---")
    multiply_kernel = f"""
        @group(0) @binding(0)
        var<storage,read> input_data: array<f32>;
        
        @group(0) @binding(1)
        var<storage,read_write> output_data: array<f32>;
        
        @compute
        @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let index = global_id.x;
            let max_elements = {tensor.size}u;
            
            if (index >= max_elements) {{
                return;
            }}
            
            // Multiply by 2.0
            output_data[index] = input_data[index] * 2.0;
        }}
    """
    
    result_bytes_mult = apply(multiply_kernel, tensor_bytes)
    result_tensor_mult = np.frombuffer(result_bytes_mult, dtype=np.float32).reshape(tensor.shape)
    
    # Verify the multiplication worked
    expected_mult = tensor * 2.0
    mult_check = np.allclose(result_tensor_mult, expected_mult)
    print(f"Multiplication check: {mult_check}")
    
    if mult_check:
        print("🎉 WebGPU kernel is working perfectly!")
        print(f"You can now process {tensor.size:,} elements on GPU!")
    else:
        print("❌ Something's still wrong with the multiplication kernel")

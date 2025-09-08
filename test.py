import numpy as np
import json
from model import WebGPUMeshNet
from tinygrad.nn.state import torch_load

def check_for_nans(data, kernel_name):
    """Check if data contains NaNs and report"""
    if isinstance(data, (bytes, memoryview)):
        # Convert bytes or memoryview to numpy array
        array = np.frombuffer(data, dtype=np.float32)
    elif isinstance(data, np.ndarray):
        array = data
    else:
        # Try to convert to numpy array
        try:
            array = np.array(data, dtype=np.float32)
        except:
            print(f"❌ {kernel_name}: Unable to convert data type {type(data)} to numpy array")
            return True, None
    
    has_nans = np.isnan(array).any()
    nan_count = np.isnan(array).sum()
    total_elements = array.size
    
    if has_nans:
        print(f"❌ {kernel_name}: FOUND {nan_count}/{total_elements} NaN values ({100*nan_count/total_elements:.2f}%)")
        # Additional diagnostics
        non_nan_values = array[~np.isnan(array)]
        if len(non_nan_values) > 0:
            print(f"   Non-NaN range: [{non_nan_values.min():.4f}, {non_nan_values.max():.4f}]")
        inf_count = np.isinf(array).sum()
        if inf_count > 0:
            print(f"   Also found {inf_count} Inf values")
    else:
        print(f"✅ {kernel_name}: No NaNs detected")
        print(f"   Value range: [{array.min():.4f}, {array.max():.4f}]")
        print(f"   Mean: {array.mean():.4f}, Std: {array.std():.4f}")
    
    return has_nans, array

def create_test_data(shape=(256, 256, 256), data_type="normal"):
    """Create different types of test data"""
    if data_type == "normal":
        # Normal distribution
        data = np.random.randn(*shape).astype(np.float32)
    elif data_type == "uniform":
        # Uniform distribution [0, 1]
        data = np.random.uniform(0, 1, shape).astype(np.float32)
    elif data_type == "large":
        # Large values that might cause overflow
        data = np.random.randn(*shape).astype(np.float32) * 100
    elif data_type == "small":
        # Very small values
        data = np.random.randn(*shape).astype(np.float32) * 0.001
    elif data_type == "mixed":
        # Mix of normal, large, and small values
        data = np.random.randn(*shape).astype(np.float32)
        # Add some extreme values
        data[::100] = np.random.randn(len(data[::100])) * 1000
        data[50::100] = np.random.randn(len(data[50::100])) * 0.0001
    elif data_type == "edge_cases":
        # Include edge cases
        data = np.random.randn(*shape).astype(np.float32)
        # Add some specific edge cases
        flat_data = data.flatten()
        flat_data[0] = 0.0
        flat_data[1] = 1e-10
        flat_data[2] = 1e10
        flat_data[3] = -1e10
        data = flat_data.reshape(shape)
    
    return data

def test_individual_kernels(config_file="model.json", model_file="model.pth"):
    """Test each kernel individually for NaN generation"""
    
    print("="*80)
    print("KERNEL NaN TESTING")
    print("="*80)
    
    # Initialize model
    print("\nInitializing WebGPU MeshNet...")
    model = WebGPUMeshNet(config_file, model_file)
    print(f"Model loaded with {len(model.weight_buffers)} weight buffers")
    
    # Test different data distributions
    test_data_types = ["normal", "uniform", "large", "small", "mixed", "edge_cases"]
    
    for data_type in test_data_types:
        print(f"\n{'='*80}")
        print(f"Testing with {data_type.upper()} distribution data")
        print(f"{'='*80}")
        
        # Create test data
        test_volume = create_test_data((256, 256, 256), data_type)
        test_bytes = test_volume.astype(np.float32).tobytes()
        
        print(f"\nInput data stats:")
        print(f"  Shape: {test_volume.shape}")
        print(f"  Range: [{test_volume.min():.4f}, {test_volume.max():.4f}]")
        print(f"  Mean: {test_volume.mean():.4f}, Std: {test_volume.std():.4f}")
        
        # Test 1: Quantile Normalization Kernel
        print(f"\n{'-'*60}")
        print("TEST 1: Quantile Normalization Kernel")
        print(f"{'-'*60}")
        try:
            norm_kernel = model.qnormalize_kernel()
            result_bytes = model.apply_kernel(norm_kernel, test_bytes)
            has_nans, result_array = check_for_nans(result_bytes, "Quantile Normalization")
        except Exception as e:
            print(f"❌ Error in Quantile Normalization: {e}")
        
        # Test 2: GELU Activation Kernel
        print(f"\n{'-'*60}")
        print("TEST 2: GELU Activation Kernel")
        print(f"{'-'*60}")
        try:
            gelu_kernel = model.gelu_kernel()
            result_bytes = model.apply_kernel(gelu_kernel, test_bytes)
            has_nans, result_array = check_for_nans(result_bytes, "GELU Activation")
            
            # Test GELU with extreme values
            extreme_data = np.array([0, 1e-10, -1e-10, 10, -10, 100, -100, 1000, -1000], dtype=np.float32)
            extreme_data_padded = np.pad(extreme_data, (0, 256*256*256 - 9), 'constant')
            extreme_bytes = extreme_data_padded.tobytes()
            print("\n  Testing GELU with extreme values...")
            result_bytes = model.apply_kernel(gelu_kernel, extreme_bytes)
            has_nans, result_array = check_for_nans(result_bytes, "GELU (extreme values)")
        except Exception as e:
            print(f"❌ Error in GELU: {e}")
        
        # Test 3: GroupNorm Kernel
        print(f"\n{'-'*60}")
        print("TEST 3: GroupNorm Kernel")
        print(f"{'-'*60}")
        try:
            # Test with different group/channel configurations
            test_configs = [(8, 8), (16, 16), (32, 32), (64, 64)]
            for num_groups, num_channels in test_configs:
                print(f"\n  Testing GroupNorm with {num_groups} groups, {num_channels} channels:")
                groupnorm_kernel = model.groupnorm_kernel(num_groups, num_channels)
                result_bytes = model.apply_kernel(groupnorm_kernel, test_bytes)
                has_nans, result_array = check_for_nans(result_bytes, f"GroupNorm({num_groups}g, {num_channels}c)")
        except Exception as e:
            print(f"❌ Error in GroupNorm: {e}")
        
        # Test 4: Dilated Conv3D Kernels
        print(f"\n{'-'*60}")
        print("TEST 4: Dilated Conv3D Kernels")
        print(f"{'-'*60}")
        
        # Load config to get layer parameters
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Test a few conv layers with different dilations
        test_layers = [0, 5, 10, 15, 20]  # Sample different layers
        
        for layer_idx in test_layers:
            if layer_idx < len(config['layers']):
                layer_config = config['layers'][layer_idx]
                dilation = layer_config.get('dilation', 1)
                padding = layer_config.get('padding', 0)
                
                print(f"\n  Testing Conv3D Layer {layer_idx} (dilation={dilation}, padding={padding}):")
                
                try:
                    conv_kernel = model.dilated_conv3d_kernel(layer_idx, dilation, padding)
                    
                    # Get corresponding weight buffer
                    weight_key = f"model.{layer_idx}.0.weight" if layer_idx < 25 else f"model.{layer_idx}.weight"
                    weight_buffer = model.weight_buffers.get(weight_key)
                    
                    if weight_buffer:
                        # Check weights for NaNs first
                        weight_shape = weight_buffer['shape']
                        print(f"    Weight shape: {weight_shape}")
                        
                        # Need to reshape input for conv layers with multiple channels
                        in_channels = weight_shape[1] if len(weight_shape) > 1 else 1
                        out_channels = weight_shape[0]
                        
                        # Create appropriate test data shape
                        if in_channels > 1:
                            test_conv_data = create_test_data((in_channels, 256, 256, 256), data_type)
                        else:
                            test_conv_data = create_test_data((256, 256, 256), data_type)
                            test_conv_data = test_conv_data.reshape(1, 256, 256, 256)
                        
                        test_conv_bytes = test_conv_data.astype(np.float32).tobytes()
                        
                        result_bytes = model.apply_kernel(conv_kernel, test_conv_bytes, weight_buffer)
                        has_nans, result_array = check_for_nans(result_bytes, f"Conv3D Layer {layer_idx}")
                    else:
                        print(f"    ⚠️  No weight buffer found for layer {layer_idx}")
                        
                except Exception as e:
                    print(f"    ❌ Error in Conv3D Layer {layer_idx}: {e}")
        
        # Test 5: Final 1x1 Conv Layer
        print(f"\n{'-'*60}")
        print("TEST 5: Final 1x1 Conv Layer (Layer 25)")
        print(f"{'-'*60}")
        try:
            conv_kernel = model.dilated_conv3d_kernel(25, 1, 0)
            weight_buffer = model.weight_buffers.get("model.25.weight")
            
            if weight_buffer:
                weight_shape = weight_buffer['shape']
                print(f"  Weight shape: {weight_shape}")
                
                # Create test data with appropriate channels
                in_channels = weight_shape[1] if len(weight_shape) > 1 else 1
                test_final_data = create_test_data((in_channels, 256, 256, 256), data_type)
                test_final_bytes = test_final_data.astype(np.float32).tobytes()
                
                result_bytes = model.apply_kernel(conv_kernel, test_final_bytes, weight_buffer)
                has_nans, result_array = check_for_nans(result_bytes, "Final 1x1 Conv")
            else:
                print("  ⚠️  No weight buffer found for final layer")
                
        except Exception as e:
            print(f"❌ Error in Final Conv: {e}")

def test_sequential_pipeline(config_file="model.json", model_file="model.pth"):
    """Test the kernels in sequence to see where NaNs first appear"""
    
    print("\n" + "="*80)
    print("SEQUENTIAL PIPELINE TESTING")
    print("="*80)
    
    model = WebGPUMeshNet(config_file, model_file)
    
    # Start with normal data
    test_volume = create_test_data((256, 256, 256), "normal")
    current_bytes = test_volume.astype(np.float32).tobytes()
    
    print("\nInitial input:")
    check_for_nans(current_bytes, "Input")
    
    # Step 1: Normalization
    print("\nStep 1: Applying normalization...")
    norm_kernel = model.qnormalize_kernel()
    current_bytes = model.apply_kernel(norm_kernel, current_bytes)
    has_nans, _ = check_for_nans(current_bytes, "After normalization")
    
    if has_nans:
        print("⚠️  NaNs detected after normalization, stopping sequential test")
        return
    
    # Step 2: First conv layer
    print("\nStep 2: Applying first conv layer...")
    layer_config = json.load(open(config_file))['layers'][0]
    conv_kernel = model.dilated_conv3d_kernel(0, layer_config['dilation'], layer_config['padding'])
    weight_buffer = model.weight_buffers.get("model.0.0.weight")
    
    if weight_buffer:
        current_bytes = model.apply_kernel(conv_kernel, current_bytes, weight_buffer)
        has_nans, _ = check_for_nans(current_bytes, "After first conv")
        
        if has_nans:
            print("⚠️  NaNs detected after first conv, stopping sequential test")
            return
    
    # Continue with more steps as needed...
    print("\n✅ Sequential test completed without NaNs in tested steps")

if __name__ == "__main__":
    # Run individual kernel tests
    test_individual_kernels()
    
    # Run sequential pipeline test
    test_sequential_pipeline()
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)

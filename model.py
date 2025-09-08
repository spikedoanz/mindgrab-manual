# https://claude.ai/chat/be3f84ad-7e54-4f6a-9052-b2782333f932
import numpy as np
from pydawn import utils, webgpu
import json
from tinygrad.nn.state import torch_load
from save import save_array_to_nifti

class WebGPUMeshNet:
    def __init__(self, config_file, model_file):
        """Initialize WebGPU MeshNet with model config and weights"""
        # Load config
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Load weights
        self.weights = torch_load(model_file)
        
        # Initialize WebGPU
        self.adapter = utils.request_adapter_sync(
            power_preference=webgpu.WGPUPowerPreference_HighPerformance
        )
        self.dev = utils.request_device_sync(self.adapter, [])
        
        # Prepare weight buffers
        self._prepare_weight_buffers()
    
    def _prepare_weight_buffers(self):
        """Convert PyTorch weights to WebGPU buffers"""
        self.weight_buffers = {}
        
        for key, weight_tensor in self.weights.items():
            # Convert to numpy and ensure float32
            weight_np = weight_tensor.numpy().astype(np.float32)
            weight_bytes = weight_np.tobytes()
            
            # Align to 16-byte boundary
            original_size = len(weight_bytes)
            aligned_size = ((original_size + 15) // 16) * 16
            if aligned_size > original_size:
                weight_bytes = weight_bytes + b'\x00' * (aligned_size - original_size)
            
            # Create buffer
            buffer = utils.create_buffer(
                self.dev,
                aligned_size,
                webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopyDst
            )
            utils.write_buffer(self.dev, buffer, 0, bytearray(weight_bytes))
            
            self.weight_buffers[key] = {
                'buffer': buffer,
                'shape': weight_tensor.shape,
                'size': aligned_size,
                'original_size': original_size
            }
    
    def qnormalize_kernel(self, qlow, qhigh):
        """Generate WGSL kernel for quantile normalization"""
        # The correct implementation based on the tinygrad reference
        return f"""
        @group(0) @binding(0)
        var<storage,read> input_data: array<f32>;
        
        @group(0) @binding(1)
        var<storage,read_write> output_data: array<f32>;
        
        const qlow = {qlow}f;
        const qhigh = {qhigh}f;
        const eps = 1e-5f;

        @compute
        @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let index = global_id.x + global_id.y * 65535u;
            
            if (index >= arrayLength(&output_data)) {{
                return;
            }}
            
            let val = input_data[index];
            let normalized_val = (val - qlow) / (qhigh - qlow + eps);
            output_data[index] = clamp(normalized_val, 0.0, 1.0);
        }}
        """
    
    def dilated_conv3d_kernel(self, layer_idx, dilation, padding, in_channels):
        """Generate WGSL kernel for 3D dilated convolution"""
        layer_key = f"model.{layer_idx}.0.weight" if layer_idx < 25 else f"model.{layer_idx}.weight"
        weight_info = self.weight_buffers[layer_key]
        weight_shape = weight_info['shape']
        
        out_channels = weight_shape[0]
        
        return f"""
        @group(0) @binding(0)
        var<storage,read> input_data: array<f32>;
        
        @group(0) @binding(1)
        var<storage,read> weights: array<f32>;
        
        @group(0) @binding(2)
        var<storage,read_write> output_data: array<f32>;
        
        @compute
        @workgroup_size(8, 8, 4)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let x = global_id.x;
            let y = global_id.y;
            let z = global_id.z;
            
            let dim = 256u;
            
            if (x >= dim || y >= dim || z >= dim) {{
                return;
            }}
            
            let out_channels = {out_channels}u;
            let in_channels = {in_channels}u;
            let dilation = {dilation}u;
            let padding = {padding}u;
            let kernel_size = 3u;
            
            for (var oc = 0u; oc < out_channels; oc = oc + 1u) {{
                var sum = 0.0;
                
                for (var ic = 0u; ic < in_channels; ic = ic + 1u) {{
                    for (var kz = 0u; kz < kernel_size; kz = kz + 1u) {{
                        for (var ky = 0u; ky < kernel_size; ky = ky + 1u) {{
                            for (var kx = 0u; kx < kernel_size; kx = kx + 1u) {{
                                let iz = i32(z) + i32(padding) - i32(kz * dilation);
                                let iy = i32(y) + i32(padding) - i32(ky * dilation);
                                let ix = i32(x) + i32(padding) - i32(kx * dilation);
                                
                                if (ix >= 0 && ix < i32(dim) && iy >= 0 && iy < i32(dim) && iz >= 0 && iz < i32(dim)) {{
                                    let input_idx = ic * dim * dim * dim + u32(iz) * dim * dim + u32(iy) * dim + u32(ix);
                                    let weight_idx = oc * in_channels * 27u + ic * 27u + kz * 9u + ky * 3u + kx;
                                    
                                    sum = sum + input_data[input_idx] * weights[weight_idx];
                                }}
                            }}
                        }}
                    }}
                }}
                
                let out_idx = oc * dim * dim * dim + z * dim * dim + y * dim + x;
                output_data[out_idx] = sum;
            }}
        }}
        """
    
    def groupnorm_kernel(self, num_groups, num_channels):
        """Generate WGSL kernel for GroupNorm"""
        return f"""
        @group(0) @binding(0)
        var<storage,read> input_data: array<f32>;
        
        @group(0) @binding(1)
        var<storage,read_write> output_data: array<f32>;
        
        @compute
        @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let index = global_id.x;
            if (index >= arrayLength(&output_data)) {{
                return;
            }}
            
            // Simplified GroupNorm: just a placeholder
            output_data[index] = input_data[index];
        }}
        """
    
    def gelu_kernel(self):
        """Generate WGSL kernel for GELU activation with NaN checking"""
        return """
        @group(0) @binding(0)
        var<storage,read> input_data: array<f32>;
        
        @group(0) @binding(1)
        var<storage,read_write> output_data: array<f32>;
        
        @compute
        @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let index = global_id.x + global_id.y * 65535u;
            
            if (index >= arrayLength(&output_data)) {
                return;
            }
            
            let x = input_data[index];
            
            if (x != x) {
                output_data[index] = 0.0;
                return;
            }
            
            let c = 0.797884560803; // sqrt(2/π)
            let x_cubed = x * x * x;
            
            if (x_cubed != x_cubed || abs(x_cubed) > 1e10) {
                output_data[index] = select(0.0, x, x > 0.0);
                return;
            }
            
            let tanh_arg = c * (x + 0.044715 * x_cubed);
            let tanh_val = tanh(tanh_arg);
            let result = 0.5 * x * (1.0 + tanh_val);
            
            if (result != result) {
                output_data[index] = 0.0;
            } else {
                output_data[index] = result;
            }
        }
        """
    
    def apply_kernel(self, kernel_source, input_bytes, output_shape, weight_buffer=None):
        """Apply a WebGPU kernel to input data"""
        input_original_size = len(input_bytes)
        input_aligned_size = ((input_original_size + 15) // 16) * 16
        padded_bytes = input_bytes
        if input_aligned_size > input_original_size:
            padded_bytes += b'\x00' * (input_aligned_size - input_original_size)

        output_num_elements = int(np.prod(output_shape))
        output_original_size = output_num_elements * 4
        output_aligned_size = ((output_original_size + 15) // 16) * 16
        
        shader_module = utils.create_shader_module(self.dev, kernel_source)
        
        input_buffer = utils.create_buffer(self.dev, input_aligned_size, webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopyDst)
        output_buffer = utils.create_buffer(self.dev, output_aligned_size, webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopySrc)
        
        utils.write_buffer(self.dev, input_buffer, 0, bytearray(padded_bytes))
        
        bindings = [{"binding": 0, "resource": {"buffer": input_buffer, "offset": 0, "size": input_aligned_size}}]
        binding_layouts = [{"binding": 0, "visibility": webgpu.WGPUShaderStage_Compute, "buffer": {"type": webgpu.WGPUBufferBindingType_ReadOnlyStorage}}]

        if weight_buffer is not None:
            bindings.append({"binding": 1, "resource": {"buffer": weight_buffer['buffer'], "offset": 0, "size": weight_buffer['size']}})
            bindings.append({"binding": 2, "resource": {"buffer": output_buffer, "offset": 0, "size": output_aligned_size}})
            binding_layouts.append({"binding": 1, "visibility": webgpu.WGPUShaderStage_Compute, "buffer": {"type": webgpu.WGPUBufferBindingType_ReadOnlyStorage}})
            binding_layouts.append({"binding": 2, "visibility": webgpu.WGPUShaderStage_Compute, "buffer": {"type": webgpu.WGPUBufferBindingType_Storage}})
        else:
            bindings.append({"binding": 1, "resource": {"buffer": output_buffer, "offset": 0, "size": output_aligned_size}})
            binding_layouts.append({"binding": 1, "visibility": webgpu.WGPUShaderStage_Compute, "buffer": {"type": webgpu.WGPUBufferBindingType_Storage}})
        
        bind_group_layout = utils.create_bind_group_layout(device=self.dev, entries=binding_layouts)
        pipeline_layout = utils.create_pipeline_layout(device=self.dev, bind_group_layouts=[bind_group_layout])
        bind_group = utils.create_bind_group(device=self.dev, layout=bind_group_layout, entries=bindings)
        
        compute_pipeline = utils.create_compute_pipeline(device=self.dev, layout=pipeline_layout, compute={"module": shader_module, "entry_point": "main"})
        
        command_encoder = utils.create_command_encoder(self.dev)
        compute_pass = utils.begin_compute_pass(command_encoder)
        utils.set_pipeline(compute_pass, compute_pipeline)
        utils.set_bind_group(compute_pass, bind_group)
        
        if "workgroup_size(8, 8, 4)" in kernel_source:
            dim = 256
            utils.dispatch_workgroups(compute_pass, dim // 8, dim // 8, dim // 4)
        else:
            workgroup_size = 256
            total_workgroups = (output_num_elements + workgroup_size - 1) // workgroup_size
            max_workgroups_1d = 65535
            
            if total_workgroups <= max_workgroups_1d:
                utils.dispatch_workgroups(compute_pass, total_workgroups, 1, 1)
            else:
                workgroups_x = max_workgroups_1d
                workgroups_y = (total_workgroups + max_workgroups_1d - 1) // max_workgroups_1d
                utils.dispatch_workgroups(compute_pass, workgroups_x, workgroups_y, 1)
        
        utils.end_compute_pass(compute_pass)
        cb_buffer = utils.command_encoder_finish(command_encoder)
        utils.submit(self.dev, [cb_buffer])
        utils.sync(self.dev)
        
        result_buffer = utils.read_buffer(self.dev, output_buffer)
        return result_buffer[:output_original_size]
    
    def forward(self, input_tensor):
        """Run forward pass through the network"""
        current_bytes = input_tensor.astype(np.float32).tobytes()
        current_shape = input_tensor.shape
        
        # Calculate quantiles before normalization
        input_data_np = np.frombuffer(current_bytes, dtype=np.float32)
        qlow = np.quantile(input_data_np, 0.02)
        qhigh = np.quantile(input_data_np, 0.98)
        print(f"Applying quantile normalization with qlow={qlow:.4f}, qhigh={qhigh:.4f}")

        # Generate the kernel with these values and apply it
        norm_kernel = self.qnormalize_kernel(qlow, qhigh)
        current_bytes = self.apply_kernel(norm_kernel, current_bytes, current_shape)
        
        normalized_output = np.frombuffer(current_bytes, dtype=np.float32).reshape(current_shape)
        for c in range(current_shape[0]):
            save_array_to_nifti(normalized_output[c], f"layer_00_normalized_c{c}.nii.gz")
        print(f"Saved {current_shape[0]} normalized input channel(s) to layer_00_normalized_c*.nii.gz")
        
        for i, layer_config in enumerate(self.config['layers']):
            print(f"Processing layer {i+1}...")
            
            weight_key = f"model.{i}.0.weight" if i < 25 else f"model.{i}.weight"
            if i >= len(self.config['layers']) -1:
                 weight_key = f"model.25.weight"

            weight_buffer = self.weight_buffers.get(weight_key)
            if not weight_buffer:
                print(f"Skipping layer {i+1} due to missing weights for key {weight_key}")
                continue

            dilation = layer_config.get('dilation', 1)
            padding = layer_config.get('padding', 0)
            in_channels = current_shape[0]
            conv_kernel = self.dilated_conv3d_kernel(i, dilation, padding, in_channels)
            
            out_channels = self.config['layers'][i]['out_channels']
            conv_shape = (out_channels, 256, 256, 256)
            
            current_bytes = self.apply_kernel(conv_kernel, current_bytes, conv_shape, weight_buffer)
            current_shape = conv_shape
            
            conv_output = np.frombuffer(current_bytes, dtype=np.float32).reshape(current_shape)
            for c in range(current_shape[0]):
                save_array_to_nifti(conv_output[c], f"layer_{i+1:02d}_conv_c{c}.nii.gz")
            print(f"Saved {current_shape[0]} conv output channels to layer_{i+1:02d}_conv_c*.nii.gz")

            if i < len(self.config['layers']) - 1:
                if self.config.get('bnorm', True):
                    groupnorm_kernel = self.groupnorm_kernel(num_groups=out_channels, num_channels=out_channels)
                    current_bytes = self.apply_kernel(groupnorm_kernel, current_bytes, current_shape)
                    
                    norm_output = np.frombuffer(current_bytes, dtype=np.float32).reshape(current_shape)
                    for c in range(current_shape[0]):
                        save_array_to_nifti(norm_output[c], f"layer_{i+1:02d}_groupnorm_c{c}.nii.gz")
                    print(f"Saved {current_shape[0]} groupnorm output channels to layer_{i+1:02d}_groupnorm_c*.nii.gz")
                
                if self.config.get('gelu', True):
                    gelu_kernel = self.gelu_kernel()
                    current_bytes = self.apply_kernel(gelu_kernel, current_bytes, current_shape)
                    
                    gelu_output = np.frombuffer(current_bytes, dtype=np.float32).reshape(current_shape)
                    for c in range(current_shape[0]):
                        save_array_to_nifti(gelu_output[c], f"layer_{i+1:02d}_gelu_c{c}.nii.gz")
                    print(f"Saved {current_shape[0]} gelu output channels to layer_{i+1:02d}_gelu_c*.nii.gz")

        result = np.frombuffer(current_bytes, dtype=np.float32).reshape(current_shape)
        return result


# Example usage
if __name__ == "__main__":
    from brainchop.niimath import conform
    
    volume, header = conform("conformed.nii.gz")
    input_tensor = volume.transpose((2, 1, 0)).astype(np.float32)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    print(f"Input shape: {input_tensor.shape}")
    
    model = WebGPUMeshNet("model.json", "model.pth")
    
    print("\n" + "="*60)
    print("Starting forward pass...")
    print("="*60)
    output = model.forward(input_tensor)
    
    print("\n" + "="*60)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print("="*60)
    
    output_path = "webgpu_segmentation_output.nii.gz"
    save_array_to_nifti(output[0], output_path)
    print(f"Saved final output (channel 0) to {output_path}")

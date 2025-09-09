import numpy as np
from pydawn import utils, webgpu
from save import save_array_to_nifti
import torch

def save_buffer(buffer, shape:tuple[int,...], fn_prefix:str="buffer"): # (c h w d)
    tensor = np.frombuffer(buffer, dtype=np.float32).reshape(shape)
    channels = shape[0]
    for c in range(channels):
        path = f"{fn_prefix}_c{c}.nii.gz"
        save_array_to_nifti(tensor[c], path)
        print("saved buffer to", path)

class SimpleWebGPUModel:
    def __init__(self, model_path):
        """Initialize WebGPU model with PyTorch weights"""
        # Load the model
        state_dict = torch.load(model_path, weights_only=False)
        if isinstance(state_dict, torch.nn.Sequential):
            state_dict = state_dict.state_dict()
        
        # Extract conv weights and biases
        self.conv_weights = {}
        self.conv_biases = {}
        
        # Parse the model structure
        for key, value in state_dict.items():
            if 'weight' in key:
                layer_idx = int(key.split('.')[0])
                self.conv_weights[layer_idx] = value.numpy().astype(np.float32)
            elif 'bias' in key:
                layer_idx = int(key.split('.')[0])
                self.conv_biases[layer_idx] = value.numpy().astype(np.float32) if value is not None else None
        
        # Initialize WebGPU
        self.adapter = utils.request_adapter_sync(
            power_preference=webgpu.WGPUPowerPreference_HighPerformance
        )
        self.dev = utils.request_device_sync(self.adapter, [])
        
        # Prepare weight buffers
        self._prepare_weight_buffers()
    
    def _prepare_weight_buffers(self):
        """Convert weights to WebGPU buffers"""
        self.weight_buffers = {}
        self.bias_buffers = {}
        
        for layer_idx, weights in self.conv_weights.items():
            # Prepare weight buffer
            weight_bytes = weights.tobytes()
            weight_size = ((len(weight_bytes) + 15) // 16) * 16
            if weight_size > len(weight_bytes):
                weight_bytes += b'\x00' * (weight_size - len(weight_bytes))
            
            buffer = utils.create_buffer(
                self.dev,
                weight_size,
                webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopyDst
            )
            utils.write_buffer(self.dev, buffer, 0, bytearray(weight_bytes))
            
            self.weight_buffers[layer_idx] = {
                'buffer': buffer,
                'shape': weights.shape,
                'size': weight_size
            }
            
            # Prepare bias buffer if exists
            if layer_idx in self.conv_biases and self.conv_biases[layer_idx] is not None:
                bias = self.conv_biases[layer_idx]
                bias_bytes = bias.tobytes()
                bias_size = ((len(bias_bytes) + 15) // 16) * 16
                if bias_size > len(bias_bytes):
                    bias_bytes += b'\x00' * (bias_size - len(bias_bytes))
                
                bias_buffer = utils.create_buffer(
                    self.dev,
                    bias_size,
                    webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopyDst
                )
                utils.write_buffer(self.dev, bias_buffer, 0, bytearray(bias_bytes))
                
                self.bias_buffers[layer_idx] = {
                    'buffer': bias_buffer,
                    'size': bias_size
                }
    
    def conv3d_kernel(self, out_channels, in_channels, kernel_size, stride, padding, dilation, has_bias=False):
        """Generate WGSL kernel for 3D convolution"""
        bias_code = """
            let bias_val = bias[oc];
            sum = sum + bias_val;
        """ if has_bias else ""
        
        return f"""
        @group(0) @binding(0)
        var<storage, read> input_data: array<f32>;
        
        @group(0) @binding(1)
        var<storage, read> weights: array<f32>;
        
        {("@group(0) @binding(2) var<storage, read> bias: array<f32>;" if has_bias else "")}
        
        @group(0) @binding({3 if has_bias else 2})
        var<storage, read_write> output_data: array<f32>;
        
        @compute
        @workgroup_size(4, 4, 4)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let x = global_id.x;
            let y = global_id.y; 
            let z = global_id.z;
            
            let in_d = 256u;
            let in_h = 256u;
            let in_w = 256u;
            let out_d = 256u;
            let out_h = 256u;
            let out_w = 256u;
            
            if (x >= out_w || y >= out_h || z >= out_d) {{
                return;
            }}
            
            let out_channels = {out_channels}u;
            let in_channels = {in_channels}u;
            let kernel_size = {kernel_size}u;
            let stride = {stride}u;
            let padding = {padding}u;
            let dilation = {dilation}u;
            
            for (var oc = 0u; oc < out_channels; oc = oc + 1u) {{
                var sum = 0.0;
                
                for (var ic = 0u; ic < in_channels; ic = ic + 1u) {{
                    for (var kd = 0u; kd < kernel_size; kd = kd + 1u) {{
                        for (var kh = 0u; kh < kernel_size; kh = kh + 1u) {{
                            for (var kw = 0u; kw < kernel_size; kw = kw + 1u) {{
                                let id = i32(z * stride) - i32(padding) + i32(kd * dilation);
                                let ih = i32(y * stride) - i32(padding) + i32(kh * dilation);
                                let iw = i32(x * stride) - i32(padding) + i32(kw * dilation);
                                
                                if (id >= 0 && id < i32(in_d) && 
                                    ih >= 0 && ih < i32(in_h) && 
                                    iw >= 0 && iw < i32(in_w)) {{
                                    
                                    let input_idx = ic * in_d * in_h * in_w + 
                                                   u32(id) * in_h * in_w + 
                                                   u32(ih) * in_w + 
                                                   u32(iw);
                                    
                                    let weight_idx = oc * in_channels * kernel_size * kernel_size * kernel_size +
                                                    ic * kernel_size * kernel_size * kernel_size +
                                                    kd * kernel_size * kernel_size +
                                                    kh * kernel_size +
                                                    kw;
                                    
                                    sum = sum + input_data[input_idx] * weights[weight_idx];
                                }}
                            }}
                        }}
                    }}
                }}
                
                {bias_code}
                
                let output_idx = oc * out_d * out_h * out_w + z * out_h * out_w + y * out_w + x;
                output_data[output_idx] = sum;
            }}
        }}
        """
    
    def relu_kernel(self):
        """Generate WGSL kernel for ReLU activation"""
        return """
        @group(0) @binding(0)
        var<storage, read_write> data: array<f32>;
        
        @compute
        @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            
            if (idx >= arrayLength(&data)) {
                return;
            }
            
            data[idx] = max(0.0, data[idx]);
        }
        """
    
    def apply_conv3d(self, input_bytes, layer_idx, conv_params):
        """Apply 3D convolution"""
        weight_buffer = self.weight_buffers[layer_idx]
        bias_buffer = self.bias_buffers.get(layer_idx)
        has_bias = bias_buffer is not None
        
        out_channels = conv_params['out_channels']
        in_channels = conv_params['in_channels']
        kernel_size = conv_params.get('kernel_size', 3)
        stride = conv_params.get('stride', 1)
        padding = conv_params['padding']
        dilation = conv_params['dilation']
        
        # Calculate output size
        output_shape = (out_channels, 256, 256, 256)
        output_size = int(np.prod(output_shape)) * 4
        output_aligned = ((output_size + 15) // 16) * 16
        
        # Prepare input buffer
        input_size = len(input_bytes)
        input_aligned = ((input_size + 15) // 16) * 16
        padded_input = input_bytes
        if input_aligned > input_size:
            padded_input += b'\x00' * (input_aligned - input_size)
        
        # Create buffers
        input_buffer = utils.create_buffer(
            self.dev, input_aligned,
            webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopyDst
        )
        output_buffer = utils.create_buffer(
            self.dev, output_aligned,
            webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopySrc
        )
        
        utils.write_buffer(self.dev, input_buffer, 0, bytearray(padded_input))
        
        # Generate kernel
        kernel_src = self.conv3d_kernel(
            out_channels, in_channels, kernel_size, 
            stride, padding, dilation, has_bias
        )
        shader = utils.create_shader_module(self.dev, kernel_src)
        
        # Setup bindings
        bindings = [
            {"binding": 0, "resource": {"buffer": input_buffer, "offset": 0, "size": input_aligned}},
            {"binding": 1, "resource": {"buffer": weight_buffer['buffer'], "offset": 0, "size": weight_buffer['size']}}
        ]
        
        entries = [
            {"binding": 0, "visibility": webgpu.WGPUShaderStage_Compute, "buffer": {"type": webgpu.WGPUBufferBindingType_ReadOnlyStorage}},
            {"binding": 1, "visibility": webgpu.WGPUShaderStage_Compute, "buffer": {"type": webgpu.WGPUBufferBindingType_ReadOnlyStorage}}
        ]
        
        if has_bias:
            bindings.append({"binding": 2, "resource": {"buffer": bias_buffer['buffer'], "offset": 0, "size": bias_buffer['size']}})
            entries.append({"binding": 2, "visibility": webgpu.WGPUShaderStage_Compute, "buffer": {"type": webgpu.WGPUBufferBindingType_ReadOnlyStorage}})
            bindings.append({"binding": 3, "resource": {"buffer": output_buffer, "offset": 0, "size": output_aligned}})
            entries.append({"binding": 3, "visibility": webgpu.WGPUShaderStage_Compute, "buffer": {"type": webgpu.WGPUBufferBindingType_Storage}})
        else:
            bindings.append({"binding": 2, "resource": {"buffer": output_buffer, "offset": 0, "size": output_aligned}})
            entries.append({"binding": 2, "visibility": webgpu.WGPUShaderStage_Compute, "buffer": {"type": webgpu.WGPUBufferBindingType_Storage}})
        
        # Create pipeline
        bind_group_layout = utils.create_bind_group_layout(self.dev, entries=entries)
        pipeline_layout = utils.create_pipeline_layout(self.dev, [bind_group_layout])
        bind_group = utils.create_bind_group(self.dev, bind_group_layout, bindings)
        pipeline = utils.create_compute_pipeline(
            self.dev, layout=pipeline_layout,
            compute={"module": shader, "entry_point": "main"}
        )
        
        # Execute
        encoder = utils.create_command_encoder(self.dev)
        compute_pass = utils.begin_compute_pass(encoder)
        utils.set_pipeline(compute_pass, pipeline)
        utils.set_bind_group(compute_pass, bind_group)
        utils.dispatch_workgroups(compute_pass, 256 // 4, 256 // 4, 256 // 4)
        utils.end_compute_pass(compute_pass)
        
        cb = utils.command_encoder_finish(encoder)
        utils.submit(self.dev, [cb])
        utils.sync(self.dev)
        
        result = utils.read_buffer(self.dev, output_buffer)
        return result[:output_size], output_shape
    
    def apply_relu(self, input_bytes):
        """Apply ReLU activation in-place"""
        size = len(input_bytes)
        aligned_size = ((size + 15) // 16) * 16
        padded = input_bytes
        if aligned_size > size:
            padded += b'\x00' * (aligned_size - size)
        
        # Create buffer
        buffer = utils.create_buffer(
            self.dev, aligned_size,
            webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopyDst | webgpu.WGPUBufferUsage_CopySrc
        )
        utils.write_buffer(self.dev, buffer, 0, bytearray(padded))
        
        # Generate and compile kernel
        kernel_src = self.relu_kernel()
        shader = utils.create_shader_module(self.dev, kernel_src)
        
        # Setup pipeline
        bind_group_layout = utils.create_bind_group_layout(self.dev, entries=[
            {"binding": 0, "visibility": webgpu.WGPUShaderStage_Compute, 
             "buffer": {"type": webgpu.WGPUBufferBindingType_Storage}}
        ])
        pipeline_layout = utils.create_pipeline_layout(self.dev, [bind_group_layout])
        bind_group = utils.create_bind_group(self.dev, bind_group_layout, [
            {"binding": 0, "resource": {"buffer": buffer, "offset": 0, "size": aligned_size}}
        ])
        pipeline = utils.create_compute_pipeline(
            self.dev, layout=pipeline_layout,
            compute={"module": shader, "entry_point": "main"}
        )
        
        # Execute
        encoder = utils.create_command_encoder(self.dev)
        compute_pass = utils.begin_compute_pass(encoder)
        utils.set_pipeline(compute_pass, pipeline)
        utils.set_bind_group(compute_pass, bind_group)
        
        num_elements = size // 4
        workgroups = (num_elements + 255) // 256
        utils.dispatch_workgroups(compute_pass, workgroups, 1, 1)
        utils.end_compute_pass(compute_pass)
        
        cb = utils.command_encoder_finish(encoder)
        utils.submit(self.dev, [cb])
        utils.sync(self.dev)
        
        result = utils.read_buffer(self.dev, buffer)
        return result[:size]
    
    def forward(self, input_tensor):
        """Run forward pass through the network"""
        # Convert input to bytes
        current_bytes = input_tensor.astype(np.float32).tobytes()
        current_shape = (1, 256, 256, 256)  # Starting shape
        
        # Define the network architecture based on your model
        layers = [
            # Conv3d(1, 5, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            {'type': 'conv', 'layer_idx': 0, 'params': {
                'out_channels': 5, 'in_channels': 1, 'padding': 1, 'dilation': 1}},
            {'type': 'relu'},
            
            # Conv3d(5, 5, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(2, 2, 2), dilation=(2, 2, 2))
            {'type': 'conv', 'layer_idx': 2, 'params': {
                'out_channels': 5, 'in_channels': 5, 'padding': 2, 'dilation': 2}},
            {'type': 'relu'},
            
            # Conv3d(5, 5, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(4, 4, 4), dilation=(4, 4, 4))
            {'type': 'conv', 'layer_idx': 4, 'params': {
                'out_channels': 5, 'in_channels': 5, 'padding': 4, 'dilation': 4}},
            {'type': 'relu'},
            
            # Conv3d(5, 5, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(8, 8, 8), dilation=(8, 8, 8))
            {'type': 'conv', 'layer_idx': 6, 'params': {
                'out_channels': 5, 'in_channels': 5, 'padding': 8, 'dilation': 8}},
            {'type': 'relu'},
            
            # Conv3d(5, 5, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(16, 16, 16), dilation=(16, 16, 16))
            {'type': 'conv', 'layer_idx': 8, 'params': {
                'out_channels': 5, 'in_channels': 5, 'padding': 16, 'dilation': 16}},
            {'type': 'relu'},
            
            # Conv3d(5, 5, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(8, 8, 8), dilation=(8, 8, 8))
            {'type': 'conv', 'layer_idx': 10, 'params': {
                'out_channels': 5, 'in_channels': 5, 'padding': 8, 'dilation': 8}},
            {'type': 'relu'},
            
            # Conv3d(5, 5, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(4, 4, 4), dilation=(4, 4, 4))
            {'type': 'conv', 'layer_idx': 12, 'params': {
                'out_channels': 5, 'in_channels': 5, 'padding': 4, 'dilation': 4}},
            {'type': 'relu'},
            
            # Conv3d(5, 5, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(2, 2, 2), dilation=(2, 2, 2))
            {'type': 'conv', 'layer_idx': 14, 'params': {
                'out_channels': 5, 'in_channels': 5, 'padding': 2, 'dilation': 2}},
            {'type': 'relu'},
            
            # Conv3d(5, 5, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            {'type': 'conv', 'layer_idx': 16, 'params': {
                'out_channels': 5, 'in_channels': 5, 'padding': 1, 'dilation': 1}},
            {'type': 'relu'},
            
            # Conv3d(5, 3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
            {'type': 'conv', 'layer_idx': 18, 'params': {
                'out_channels': 3, 'in_channels': 5, 'padding': 0, 'dilation': 1, 
                'kernel_size': 1}}
        ]
        
        # Process each layer
        for i, layer in enumerate(layers):
            if layer['type'] == 'conv':
                print(f"Applying Conv3d layer {layer['layer_idx']}...")
                current_bytes, current_shape = self.apply_conv3d(
                    current_bytes, 
                    layer['layer_idx'],
                    layer['params']
                )
            elif layer['type'] == 'relu':
                print(f"Applying ReLU...")
                current_bytes = self.apply_relu(current_bytes)
            prefix = f"layer_{i}_{layer['type']}"
            save_buffer(current_bytes, current_shape, prefix)
        
        # Convert back to numpy array
        result = np.frombuffer(current_bytes, dtype=np.float32).reshape(current_shape)
        return result



# Example usage
if __name__ == "__main__":
    from brainchop.niimath import conform
    
    # Load and prepare the NIfTI file
    print("Loading NIfTI file...")
    volume, header = conform("conformed.nii.gz")
    input_tensor = volume.transpose((2, 1, 0)).astype(np.float32)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Input range: [{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
    
    # Initialize model
    print("\nInitializing WebGPU model...")
    model = SimpleWebGPUModel("weights/tissue_fast/model.pth")
    
    # Run inference
    print("\n" + "="*60)
    print("Starting forward pass...")
    print("="*60 + "\n")
    
    output = model.forward(input_tensor)
    
    print("\n" + "="*60)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print("="*60)
    
    # Save final output
    print(output.shape)
    save_buffer(np.frombuffer(output), output.shape, "tissue_fast_output")

```
# load_model.py
from tinygrad.nn.state import torch_load

state_dict = torch_load("model.pth")
print("Available keys:", list(state_dict.keys()))
```



```
# model.json
{
  "header": "A model architecture for 256^3 brains with 5 decoders, where dilation in each is structured as 16->8->4->2->1",
  "bnorm": true,
  "gelu": true,
  "dropout_p": 0,
  "layers": [
    {
      "in_channels": -1,
      "out_channels": 5,
      "kernel_size": 3,
      "padding": 16,
      "stride": 1,
      "dilation": 16
    },
    {
      "in_channels": 5,
      "out_channels": 5,
      "kernel_size": 3,
      "padding": 8,
      "stride": 1,
      "dilation": 8
    },
    {
      "in_channels": 5,
      "out_channels": 5,
      "kernel_size": 3,
      "padding": 4,
      "stride": 1,
      "dilation": 4
    },
    {
      "in_channels": 5,
      "out_channels": 5,
      "kernel_size": 3,
      "padding": 2,
      "stride": 1,
      "dilation": 2
    },
    {
      "in_channels": 5,
      "out_channels": 5,
      "kernel_size": 3,
      "padding": 1,
      "stride": 1,
      "dilation": 1
    },
    {
      "in_channels": 5,
      "out_channels": 5,
      "kernel_size": 3,
      "padding": 16,
      "stride": 1,
      "dilation": 16
    },
    {
      "in_channels": 5,
      "out_channels": 5,
      "kernel_size": 3,
      "padding": 8,
      "stride": 1,
      "dilation": 8
    },
    {
      "in_channels": 5,
      "out_channels": 5,
      "kernel_size": 3,
      "padding": 4,
      "stride": 1,
      "dilation": 4
    },
    {
      "in_channels": 5,
      "out_channels": 5,
      "kernel_size": 3,
      "padding": 2,
      "stride": 1,
      "dilation": 2
    },
    {
      "in_channels": 5,
      "out_channels": 5,
      "kernel_size": 3,
      "padding": 1,
      "stride": 1,
      "dilation": 1
    },
    {
      "in_channels": 5,
      "out_channels": 5,
      "kernel_size": 3,
      "padding": 16,
      "stride": 1,
      "dilation": 16
    },
    {
      "in_channels": 5,
      "out_channels": 5,
      "kernel_size": 3,
      "padding": 8,
      "stride": 1,
      "dilation": 8
    },
    {
      "in_channels": 5,
      "out_channels": 5,
      "kernel_size": 3,
      "padding": 4,
      "stride": 1,
      "dilation": 4
    },
    {
      "in_channels": 5,
      "out_channels": 5,
      "kernel_size": 3,
      "padding": 2,
      "stride": 1,
      "dilation": 2
    },
    {
      "in_channels": 5,
      "out_channels": 5,
      "kernel_size": 3,
      "padding": 1,
      "stride": 1,
      "dilation": 1
    },
    {
      "in_channels": 5,
      "out_channels": 5,
      "kernel_size": 3,
      "padding": 16,
      "stride": 1,
      "dilation": 16
    },
    {
      "in_channels": 5,
      "out_channels": 5,
      "kernel_size": 3,
      "padding": 8,
      "stride": 1,
      "dilation": 8
    },
    {
      "in_channels": 5,
      "out_channels": 5,
      "kernel_size": 3,
      "padding": 4,
      "stride": 1,
      "dilation": 4
    },
    {
      "in_channels": 5,
      "out_channels": 5,
      "kernel_size": 3,
      "padding": 2,
      "stride": 1,
      "dilation": 2
    },
    {
      "in_channels": 5,
      "out_channels": 5,
      "kernel_size": 3,
      "padding": 1,
      "stride": 1,
      "dilation": 1
    },
    {
      "in_channels": 5,
      "out_channels": 5,
      "kernel_size": 3,
      "padding": 16,
      "stride": 1,
      "dilation": 16
    },
    {
      "in_channels": 5,
      "out_channels": 5,
      "kernel_size": 3,
      "padding": 8,
      "stride": 1,
      "dilation": 8
    },
    {
      "in_channels": 5,
      "out_channels": 5,
      "kernel_size": 3,
      "padding": 4,
      "stride": 1,
      "dilation": 4
    },
    {
      "in_channels": 5,
      "out_channels": 5,
      "kernel_size": 3,
      "padding": 2,
      "stride": 1,
      "dilation": 2
    },
    {
      "in_channels": 5,
      "out_channels": 5,
      "kernel_size": 3,
      "padding": 1,
      "stride": 1,
      "dilation": 1
    },
    {
      "in_channels": 5,
      "out_channels": -1,
      "kernel_size": 1,
      "padding": 0,
      "stride": 1,
      "dilation": 1
    }
  ]
}
```



```
$ python load_model.py
Available keys: ['model.0.0.weight', 'model.1.0.weight', 'model.2.0.weight', 'model.3.0.weight', 'model.4.0.weight', 'model.5.0.weight', 'model.6.0.weight', 'model.7.0.weight', 'model.8.0.weight', 'model.9.0.weight', 'model.10.0.weight', 'model.11.0.weight', 'model.12.0.weight', 'model.13.0.weight', 'model.14.0.weight', 'model.15.0.weight', 'model.16.0.weight', 'model.17.0.weight', 'model.18.0.weight', 'model.19.0.weight', 'model.20.0.weight', 'model.21.0.weight', 'model.22.0.weight', 'model.23.0.weight', 'model.24.0.weight', 'model.25.weight', 'model.25.bias']
```


```
# identity.py
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
```


```
# tiny_meshnet.py
from tinygrad import Tensor, nn, dtypes
from tinygrad.nn.state import torch_load, load_state_dict
import json
import numpy as np
import time


def convert_keys(torch_state_dict, tiny_state_dict):
    torch_keys = torch_state_dict.keys()
    tiny_keys = tiny_state_dict.keys()
    new_dict = {}
    for f, t in zip(torch_keys, tiny_keys):
        new_dict[t] = torch_state_dict[f]
    return new_dict


def qnormalize(img: Tensor, qmin=0.02, qmax=0.98) -> Tensor:
    """Unit interval preprocessing with clipping"""
    img = img.numpy()
    qlow = np.quantile(img, qmin)
    qhigh = np.quantile(img, qmax)
    img = (img - qlow) / (qhigh - qlow)
    img = np.clip(img, 0, 1)  # Clip the values to be between 0 and 1
    return Tensor(img)


def set_channel_num(config, in_channels, n_classes, channels):
    # input layer
    config["layers"][0]["in_channels"] = in_channels
    config["layers"][0]["out_channels"] = channels
    # output layer
    config["layers"][-1]["in_channels"] = channels
    config["layers"][-1]["out_channels"] = n_classes
    # hidden layers
    for layer in config["layers"][1:-1]:
        layer["in_channels"] = layer["out_channels"] = channels
    return config


def construct_layer(dropout_p=0, bnorm=True, gelu=False, *args, **kwargs):
    layers = []
    kwargs["kernel_size"] = [kwargs["kernel_size"]] * 3
    layers.append(nn.Conv2d(*args, **kwargs))
    if bnorm:
        layers.append(
            nn.GroupNorm(
                num_groups=kwargs["out_channels"],
                num_channels=kwargs["out_channels"],
                affine=False,
            )
        )

    relu = lambda x: x.relu()
    gelu = lambda x: x.gelu()
    dropout = lambda x: x.dropout(dropout_p)
    layers.append(gelu if gelu else relu)
    if dropout_p > 0:
        layers.append(dropout)
    return layers


class MeshNet:
    """Configurable MeshNet from https://arxiv.org/pdf/1612.00940.pdf"""

    def __init__(self, in_channels, n_classes, channels, config_file, fat=None):
        """Init"""
        with open(config_file, "r") as f:
            config = set_channel_num(json.load(f), in_channels, n_classes, channels)
        if fat is not None:
            chn = int(channels * 1.5)
            if fat in {"i", "io"}:
                config["layers"][0]["out_channels"] = chn
                config["layers"][1]["in_channels"] = chn
            if fat == "io":
                config["layers"][-1]["in_channels"] = chn
                config["layers"][-2]["out_channels"] = chn
            if fat == "b":
                config["layers"][3]["out_channels"] = chn
                config["layers"][4]["in_channels"] = chn

        self.model = []
        for block_kwargs in config["layers"][:-1]:  # All but the last layer
            self.model.extend(
                construct_layer(
                    dropout_p=config["dropout_p"],
                    bnorm=config["bnorm"],
                    gelu=config["gelu"],
                    **{**block_kwargs, "bias": False},  # middle layers have no bias
                )
            )

        # Handle last layer specially - add it to model list
        last_config = config["layers"][-1]
        self.model.append(
            nn.Conv2d(
                last_config["in_channels"],
                last_config["out_channels"],
                kernel_size=[last_config["kernel_size"]] * 3,
                padding=last_config["padding"],
                stride=last_config["stride"],
                dilation=last_config["dilation"],
                bias=False,  # Enable bias in the conv layer
            )
        )

    def __call__(self, x):
        x = qnormalize(x)  # TODO: interpret normalization from config file
        for layer in self.model:
            x = layer(x)
        return x


def load_meshnet(
    config_fn: str,
    model_fn: str,
    in_channels: int = 1,
    channels: int = 15,
    out_channels: int = 2,
):
    # TODO: Interpret channel info from config
    model = MeshNet(
        in_channels=in_channels,
        n_classes=out_channels,
        channels=channels,
        config_file=config_fn,
    )
    state_dict = torch_load(model_fn)
    state_dict = convert_keys(state_dict, nn.state.get_state_dict(model))
    load_state_dict(model, state_dict, strict=True)
    return model
```

i'm trying to hand write a webgpu implementation of the following network.

the compiler approaches i've been using have not been working. it has something
to do with the fact that webgpu has a threadgroup count limitation, which can
fortunately be bypassed by chunking either the input or the intermediate
computation.

above, i've attached some stuff for:
1. using pydawn and running a kernel on an input of size 256^3
2. loading a model
3. outputs of the load scripts (which contain model weight descriptions)
4. the model layout itself (which details how the convs are constructed)
5. the implementation of the model in tinygrad

could you help me write some kernels to implement this model in webgpu? there
are only a handlful of operations in the entire model:
1. dilated convs with varying dilations.
2. groupnorm

the rest are simple activation functions that are very straightforward to
implement.

important note: all of the intermediate tensors in between layers will be also
of size (1,1,256,256,256). i'm not entirely sure what implications this has for 
how to construct the kenrels, but hopefully this helps.

love,
spike

```
# model.py
# https://claude.ai/chat/be3f84ad-7e54-4f6a-9052-b2782333f932
import numpy as np
from pydawn import utils, webgpu
import json
from tinygrad.nn.state import torch_load

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
    
    def qnormalize_kernel(self):
        """Generate WGSL kernel for quantile normalization"""
        # This is a simplified version - full implementation would need multiple passes
        return """
        @group(0) @binding(0)
        var<storage,read> input_data: array<f32>;
        
        @group(0) @binding(1)
        var<storage,read_write> output_data: array<f32>;
        
        @compute
        @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let index = global_id.x + global_id.y * 65535u;
            let max_elements = 16777216u; // 256^3
            
            if (index >= max_elements) {
                return;
            }
            
            // Simplified normalization - you'd need to compute quantiles in a separate pass
            let val = input_data[index];
            // For now, just clamp to [0,1] - replace with proper quantile normalization
            output_data[index] = clamp(val, 0.0, 1.0);
        }
        """
    
    def dilated_conv3d_kernel(self, layer_idx, dilation, padding):
        """Generate WGSL kernel for 3D dilated convolution"""
        layer_key = f"model.{layer_idx}.0.weight" if layer_idx < 25 else f"model.{layer_idx}.weight"
        weight_info = self.weight_buffers[layer_key]
        weight_shape = weight_info['shape']
        
        # For 3D conv: weight shape is [out_channels, in_channels, D, H, W]
        # Since kernel_size is always 3, D=H=W=3
        out_channels = weight_shape[0]
        in_channels = weight_shape[1] if len(weight_shape) > 1 else 1
        
        kernel = f"""
        @group(0) @binding(0)
        var<storage,read> input_data: array<f32>;
        
        @group(0) @binding(1)
        var<storage,read> weights: array<f32>;
        
        @group(0) @binding(2)
        var<storage,read_write> output_data: array<f32>;
        
        @compute
        @workgroup_size(8, 8, 4)  // Smaller workgroup for 3D
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let x = global_id.x;
            let y = global_id.y;
            let z = global_id.z;
            
            // Volume dimensions
            let dim = 256u;
            
            if (x >= dim || y >= dim || z >= dim) {{
                return;
            }}
            
            let out_channels = {out_channels}u;
            let in_channels = {in_channels}u;
            let dilation = {dilation}u;
            let padding = {padding}u;
            let kernel_size = 3u;
            
            // For each output channel
            for (var oc = 0u; oc < out_channels; oc = oc + 1u) {{
                var sum = 0.0;
                
                // Convolution operation
                for (var ic = 0u; ic < in_channels; ic = ic + 1u) {{
                    for (var kz = 0u; kz < kernel_size; kz = kz + 1u) {{
                        for (var ky = 0u; ky < kernel_size; ky = ky + 1u) {{
                            for (var kx = 0u; kx < kernel_size; kx = kx + 1u) {{
                                // Calculate dilated positions
                                let dz = z + padding - kz * dilation;
                                let dy = y + padding - ky * dilation;
                                let dx = x + padding - kx * dilation;
                                
                                // Check bounds
                                if (dx < dim && dy < dim && dz < dim) {{
                                    // Input index: [batch=0, channel=ic, z, y, x]
                                    let input_idx = ic * dim * dim * dim + dz * dim * dim + dy * dim + dx;
                                    
                                    // Weight index: [oc, ic, kz, ky, kx]
                                    let weight_idx = oc * in_channels * 27u + ic * 27u + kz * 9u + ky * 3u + kx;
                                    
                                    sum = sum + input_data[input_idx] * weights[weight_idx];
                                }}
                            }}
                        }}
                    }}
                }}
                
                // Output index: [batch=0, channel=oc, z, y, x]
                let out_idx = oc * dim * dim * dim + z * dim * dim + y * dim + x;
                output_data[out_idx] = sum;
            }}
        }}
        """
        
        return kernel
    
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
            let index = global_id.x + global_id.y * 65535u;
            let dim = 256u;
            let max_elements = dim * dim * dim;
            
            if (index >= max_elements) {{
                return;
            }}
            
            let num_channels = {num_channels}u;
            let num_groups = {num_groups}u;
            let channels_per_group = num_channels / num_groups;
            
            // Determine which channel this element belongs to
            let elements_per_channel = max_elements / num_channels;
            let channel = index / elements_per_channel;
            let group = channel / channels_per_group;
            
            // Compute mean and variance for this group
            // This is simplified - in practice you'd need multiple passes
            var mean = 0.0;
            var variance = 0.0;
            var count = 0u;
            
            // Calculate group statistics (simplified)
            let group_start = group * channels_per_group * elements_per_channel;
            let group_end = (group + 1u) * channels_per_group * elements_per_channel;
            
            for (var i = group_start; i < group_end; i = i + 1u) {{
                mean = mean + input_data[i];
                count = count + 1u;
            }}
            mean = mean / f32(count);
            
            for (var i = group_start; i < group_end; i = i + 1u) {{
                let diff = input_data[i] - mean;
                variance = variance + diff * diff;
            }}
            variance = variance / f32(count);
            
            // Apply normalization
            let eps = 1e-5;
            let normalized = (input_data[index] - mean) / sqrt(variance + eps);
            output_data[index] = normalized;
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
            let max_elements = 16777216u; // 256^3
            
            if (index >= max_elements) {
                return;
            }
            
            let x = input_data[index];
            
            // Check for NaN
            if (x != x) {
                output_data[index] = 0.0;
                return;
            }
            
            // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            let c = 0.797884560803; // sqrt(2/π)
            let x_cubed = x * x * x;
            
            // Check for overflow in x^3
            if (x_cubed != x_cubed || abs(x_cubed) > 1e10) {
                output_data[index] = select(0.0, x, x > 0.0);  // Saturate to 0 or x
                return;
            }
            
            let tanh_arg = c * (x + 0.044715 * x_cubed);
            let tanh_val = tanh(tanh_arg);
            let result = 0.5 * x * (1.0 + tanh_val);
            
            // Final NaN check
            if (result != result) {
                output_data[index] = 0.0;
            } else {
                output_data[index] = result;
            }
        }
        """
    
    def apply_kernel(self, kernel_source, input_bytes, weight_buffer=None):
        """Apply a WebGPU kernel to input data"""
        original_size = len(input_bytes)
        num_elements = original_size // 4  # 4 bytes per float32
        
        # Align buffer size
        aligned_size = ((original_size + 15) // 16) * 16
        if aligned_size > original_size:
            padded_bytes = input_bytes + b'\x00' * (aligned_size - original_size)
        else:
            padded_bytes = input_bytes
        
        # Create shader module
        shader_module = utils.create_shader_module(self.dev, kernel_source)
        
        # Create buffers
        input_buffer = utils.create_buffer(
            self.dev,
            aligned_size,
            webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopyDst
        )
        output_buffer = utils.create_buffer(
            self.dev,
            aligned_size,
            webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopySrc
        )
        
        utils.write_buffer(self.dev, input_buffer, 0, bytearray(padded_bytes))
        
        # Setup bindings
        bindings = [
            {
                "binding": 0,
                "resource": {"buffer": input_buffer, "offset": 0, "size": aligned_size},
            },
        ]
        
        if weight_buffer is not None:
            bindings.append({
                "binding": 1,
                "resource": {"buffer": weight_buffer['buffer'], "offset": 0, "size": weight_buffer['size']},
            })
            bindings.append({
                "binding": 2,
                "resource": {"buffer": output_buffer, "offset": 0, "size": aligned_size},
            })
        else:
            bindings.append({
                "binding": 1,
                "resource": {"buffer": output_buffer, "offset": 0, "size": aligned_size},
            })
        
        # Create bind group layout entries
        binding_layouts = [
            {
                "binding": 0,
                "visibility": webgpu.WGPUShaderStage_Compute,
                "buffer": {"type": webgpu.WGPUBufferBindingType_ReadOnlyStorage},
            },
        ]
        
        if weight_buffer is not None:
            binding_layouts.append({
                "binding": 1,
                "visibility": webgpu.WGPUShaderStage_Compute,
                "buffer": {"type": webgpu.WGPUBufferBindingType_ReadOnlyStorage},
            })
            binding_layouts.append({
                "binding": 2,
                "visibility": webgpu.WGPUShaderStage_Compute,
                "buffer": {"type": webgpu.WGPUBufferBindingType_Storage},
            })
        else:
            binding_layouts.append({
                "binding": 1,
                "visibility": webgpu.WGPUShaderStage_Compute,
                "buffer": {"type": webgpu.WGPUBufferBindingType_Storage},
            })
        
        # Create pipeline
        bind_group_layout = utils.create_bind_group_layout(device=self.dev, entries=binding_layouts)
        pipeline_layout = utils.create_pipeline_layout(device=self.dev, bind_group_layouts=[bind_group_layout])
        bind_group = utils.create_bind_group(device=self.dev, layout=bind_group_layout, entries=bindings)
        
        compute_pipeline = utils.create_compute_pipeline(
            device=self.dev,
            layout=pipeline_layout,
            compute={"module": shader_module, "entry_point": "main"},
        )
        
        # Run compute pass
        command_encoder = utils.create_command_encoder(self.dev)
        compute_pass = utils.begin_compute_pass(command_encoder)
        utils.set_pipeline(compute_pass, compute_pipeline)
        utils.set_bind_group(compute_pass, bind_group)
        
        # For 3D workgroups
        if weight_buffer is not None:  # Conv3D uses 3D dispatch
            dim = 256
            workgroup_size = 8
            workgroups = (dim + workgroup_size - 1) // workgroup_size
            utils.dispatch_workgroups(compute_pass, workgroups, workgroups, workgroups // 2)
        else:  # Use 2D dispatch for element-wise ops
            workgroup_size = 256
            total_workgroups = (num_elements + workgroup_size - 1) // workgroup_size
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
        
        # Read result
        result_buffer = utils.read_buffer(self.dev, output_buffer)
        return result_buffer[:original_size]
    
    def forward(self, input_tensor):
        """Run forward pass through the network"""
        # Convert input to bytes
        current_bytes = input_tensor.astype(np.float32).tobytes()
        
        # Apply normalization
        norm_kernel = self.qnormalize_kernel()
        current_bytes = self.apply_kernel(norm_kernel, current_bytes)
        
        # Process each layer according to config
        for i, layer_config in enumerate(self.config['layers']):
            print(f"Processing layer {i}...")
            
            if i < len(self.config['layers']) - 1:
                # Conv3D layer
                dilation = layer_config['dilation']
                padding = layer_config['padding']
                conv_kernel = self.dilated_conv3d_kernel(i, dilation, padding)
                
                # Get weight buffer for this layer
                weight_key = f"model.{i}.0.weight"
                weight_buffer = self.weight_buffers.get(weight_key)
                
                if weight_buffer:
                    current_bytes = self.apply_kernel(conv_kernel, current_bytes, weight_buffer)
                    
                    # Apply GroupNorm if enabled
                    if self.config.get('bnorm', True):
                        groupnorm_kernel = self.groupnorm_kernel(
                            num_groups=layer_config['out_channels'],
                            num_channels=layer_config['out_channels']
                        )
                        current_bytes = self.apply_kernel(groupnorm_kernel, current_bytes)
                    
                    # Apply activation (GELU or ReLU)
                    if self.config.get('gelu', True):
                        gelu_kernel = self.gelu_kernel()
                        current_bytes = self.apply_kernel(gelu_kernel, current_bytes)
            else:
                # Final 1x1 conv layer
                conv_kernel = self.dilated_conv3d_kernel(25, 1, 0)  # No dilation/padding for 1x1
                weight_buffer = self.weight_buffers.get("model.25.weight")
                if weight_buffer:
                    current_bytes = self.apply_kernel(conv_kernel, current_bytes, weight_buffer)
        
        # Convert back to numpy
        result_shape = (self.config['layers'][-1]['out_channels'], 256, 256, 256)
        result = np.frombuffer(current_bytes, dtype=np.float32).reshape(result_shape)
        return result


# Example usage
if __name__ == "__main__":
    from brainchop.niimath import conform
    
    # Load input
    volume, header = conform("conformed.nii.gz")
    input_tensor = volume.transpose((2, 1, 0)).astype(np.float32)
    print(f"Input shape: {input_tensor.shape}")
    
    # Initialize model
    model = WebGPUMeshNet("model.json", "model.pth")
    
    # Run inference with NaN checking
    print("\n" + "="*60)
    print("Starting forward pass with NaN checking...")
    print("="*60)
    output = model.forward(input_tensor)
    
    print("\n" + "="*60)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print("="*60)
    
    # Save to NIfTI file
    output_path = "webgpu_segmentation_output.nii.gz"
    model.save_to_nifti(
        output, 
        header, 
        output_path,
        inverse_conform_path="conformed.nii.gz"  # Optional: for inverse conforming
    )
    
    # Optionally export class probabilities
    # model.export_class_probabilities(output, output_path)
```

```
# cli.py
# this is an example of how to save a model output to nifti
import os
import argparse
import subprocess
import json
from pathlib import Path

import numpy as np
from tinygrad.tensor import Tensor
from brainchop.niimath import (
    conform,
    set_header_intent_label,
    bwlabel,
    grow_border,
    niimath_dtype,
)

from brainchop.utils import (
    update_models,
    list_models,
    get_model,
    export_classes,
    AVAILABLE_MODELS,
    cleanup,
    crop_to_cutoff,
    pad_to_original_size,
)


def load_optimization_cache(model_name):
    """
    Load optimization cache for a given model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        dict: Optimization cache data with 'beams' list, or empty structure if not found
    """
    cache_dir = Path.home() / ".cache" / "brainchop" / "models" / model_name
    cache_file = cache_dir / "optimizations.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # If file is corrupted, return empty structure
            pass
    
    return {"beams": []}


def save_optimization_cache(model_name, batch_size, beam_value):
    """
    Save optimization data to cache.
    
    Args:
        model_name: Name of the model
        batch_size: Batch size used
        beam_value: BEAM value that was successful
    """
    cache_dir = Path.home() / ".cache" / "brainchop" / "models" / model_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "optimizations.json"
    
    # Load existing cache
    cache_data = load_optimization_cache(model_name)
    
    # Check if this BS/BEAM combination already exists
    for entry in cache_data["beams"]:
        if entry["BS"] == batch_size and entry["BEAM"] == beam_value:
            return  # Already cached
    
    # Add new entry
    cache_data["beams"].append({"BS": batch_size, "BEAM": beam_value})
    
    # Sort by batch size for easier reading
    cache_data["beams"].sort(key=lambda x: (x["BS"], x["BEAM"]))
    
    # Save to file
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)


def get_best_beam_for_batch_size(model_name, batch_size):
    """
    Get the best (largest) BEAM value for a given batch size.
    
    Args:
        model_name: Name of the model
        batch_size: Current batch size
        
    Returns:
        int: Best BEAM value for this batch size, or None if not found
    """
    cache_data = load_optimization_cache(model_name)
    
    # Find all BEAM values for this batch size
    beam_values = [entry["BEAM"] for entry in cache_data["beams"] if entry["BS"] == batch_size]
    
    if beam_values:
        return max(beam_values)
    
    return None


def is_first_run(model_name, batch_size):
    """
    Check if this is the first run for a given model and batch size.
    
    Args:
        model_name: Name of the model
        batch_size: Batch size to check
        
    Returns:
        bool: True if no optimization exists for this model/batch_size combo
    """
    cache_data = load_optimization_cache(model_name)
    
    # Check if any optimization exists for this batch size
    for entry in cache_data["beams"]:
        if entry["BS"] == batch_size:
            return False
    
    return True


def preoptimize(model_name, beam, batch_size=1):
    """
    Pre-optimize a model by running it with a random input tensor and specified BEAM value.
    
    Args:
        model_name: Name of the model to optimize
        beam: BEAM optimization value to use
        batch_size: Batch size for the input tensor (default: 1)
    """
    print(f"brainchop :: Pre-optimizing model '{model_name}' with BEAM={beam}, BS={batch_size}...")
    print(f"brainchop :: This may take a few moments for the initial compilation...")
    
    # Store original BEAM value
    original_beam = os.environ.get("BEAM")
    
    try:
        # Set BEAM environment variable for optimization
        os.environ["BEAM"] = str(beam)
        
        # Load the model with the specified BEAM value
        model = get_model(model_name)
        
        # Generate random input tensor with shape (BS, 1, 256, 256, 256)
        random_input = np.random.randn(batch_size, 1, 256, 256, 256).astype(np.float32)
        input_tensor = Tensor(random_input)
        
        print(f"brainchop :: Running optimization pass...")
        
        # Run inference to trigger compilation/optimization
        output = model(input_tensor)
        
        # Force computation to complete (realize the tensor)
        output.realize()
        
        print(f"brainchop :: Pre-optimization complete! Model is now optimized for BS={batch_size}")
        
        # Save this optimization to cache
        save_optimization_cache(model_name, batch_size, beam)
        
        return True
        
    except Exception as e:
        print(f"brainchop :: Pre-optimization failed: {e}")
        return False
        
    finally:
        # Restore original BEAM environment variable
        if original_beam is not None:
            os.environ["BEAM"] = original_beam
        elif "BEAM" in os.environ:
            del os.environ["BEAM"]


def prompt_for_optimization(model_name, batch_size):
    """
    Prompt user to optimize the model on first run.
    
    Args:
        model_name: Name of the model
        batch_size: Batch size to optimize for
        
    Returns:
        bool: True if optimization was performed successfully, False otherwise
    """
    print(f"\nbrainchop :: First run detected for model '{model_name}' with batch size {batch_size}")
    print(f"brainchop :: Would you like to pre-optimize the model for faster subsequent runs?")
    print(f"brainchop :: This will compile the model with BEAM=2 optimization (recommended)")
    
    while True:
        response = input("brainchop :: Optimize now? [y/n]: ").strip().lower()
        
        if response == 'y':
            return preoptimize(model_name, beam=2, batch_size=batch_size)
        elif response == 'n':
            print("brainchop :: Skipping optimization. Proceeding with unoptimized model...")
            return False
        else:
            print("brainchop :: Please enter 'y' for yes or 'n' for no")


def generate_output_filename(input_path, modelname, index, output_dir=None):
    """
    Generate output filename based on input filename, model name, and index.
    
    Args:
        input_path: Path to input file
        modelname: Name of the segmentation model used
        index: Processing index/order
        output_dir: Optional output directory (uses current dir if None)
        
    Returns:
        str: Generated output filename in format {input_name}_{modelname}_output_{index}.nii.gz
    """
    input_file = Path(input_path)
    
    # Extract base name without extensions (.nii.gz or .nii)
    base_name = input_file.name
    if base_name.endswith('.nii.gz'):
        base_name = base_name[:-7]  # Remove .nii.gz
    elif base_name.endswith('.nii'):
        base_name = base_name[:-4]  # Remove .nii
    else:
        # Remove any extension for non-nii files
        base_name = input_file.stem
    
    # Generate output filename with model name and index
    output_filename = f"{base_name}_{modelname}_output_{index}.nii.gz"
    
    # Use output directory if specified, otherwise use current directory
    if output_dir:
        output_path = Path(output_dir) / output_filename
    else:
        output_path = Path(output_filename)
    
    return str(output_path.absolute())


def get_parser():
    parser = argparse.ArgumentParser(
        description="BrainChop: portable brain segmentation tool"
    )
    parser.add_argument("input", nargs="*", help="Input NIfTI file path(s)")
    parser.add_argument(
        "-l", "--list", action="store_true", help="List available models"
    )
    parser.add_argument(
        "-i",
        "--inverse-conform",
        action="store_true",
        help="Perform inverse conformation into original image space",
    )
    parser.add_argument(
        "-u", "--update", action="store_true", help="Update the model listing"
    )
    parser.add_argument(
        "-o", "--output", default="output.nii.gz", help="Output NIfTI file path (for single input) or output directory (for multiple inputs)"
    )
    parser.add_argument(
        "-a",
        "--mask",
        nargs="?",  # 0 or 1 arguments
        const="mask.nii.gz",  # if they just say `--mask` with no value
        default=None,  # if they don't mention `--mask` at all
        help="If provided and using mindgrab, write out the mask (defaults to mask.nii.gz when used without a value)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=next(iter(AVAILABLE_MODELS.keys())),
        help=f"Name of segmentation model, default: {next(iter(AVAILABLE_MODELS.keys()))}",
    )
    parser.add_argument(
        "-c",
        "--custom",
        type=str,
        help="Path to custom model directory (model.json and model.bin)",
    )
    parser.add_argument(
        "--comply",
        action="store_true",
        default=False,
        help="Insert compliance arguments to `niimath` before '-conform'",
    )
    parser.add_argument(
        "--ct",
        action="store_true",
        default=False,
        help="Convert CT scans from 'Hounsfield' to 'Cormack' units to emphasize soft tissue contrast",
    )
    parser.add_argument(
        "--crop",
        nargs="?",  # 0 or 1 arguments
        type=float,
        const=2,  # if they just say `--crop` with no value
        default=False,  # if they don't mention `--crop` at all
        help="Crop the input for faster execution. May reduce accuracy.(defaults to percentile 2 cutoff)",
    )
    parser.add_argument(
        "-ss",
        "--skull-strip",
        action="store_true",
        help="Return just the brain compartment. An alias for -m mindgrab, that overrides -m parameter",
    )
    parser.add_argument(
        "-ec",
        "--export-classes",
        action="store_true",
        help="Export class probability maps",
    )
    parser.add_argument(
        "-b",
        "--border",
        type=int,
        default=0,
        help="Mask border threshold in mm. Default is 0. Makes a difference only if the model is `mindgrab`",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing multiple inputs (default: 1)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip the optimization prompt on first run",
    )
    return parser


def preprocess_input(args):
    """
    Handle input preprocessing: loading, conforming, and cropping.
    
    Returns:
        tuple: (image_tensor, volume, header, crop_coords)
    """
    # Load and conform input volume
    volume, header = conform(args.input, comply=args.comply, ct=args.ct)
    crop_coords = None
    
    # Apply cropping if requested
    if args.crop:
        volume, crop_coords = crop_to_cutoff(volume, args.crop)
        print(f"brainchop :: cropped to {volume.shape}")
    
    # Convert to tensor format expected by model
    image = Tensor(volume.transpose((2, 1, 0)).astype(np.float32)).rearrange(
        "... -> 1 1 ..."
    )
    
    return image, volume, header, crop_coords


def preprocess_batch(input_files, args):
    """
    Handle batch preprocessing: loading, conforming, and cropping multiple inputs.
    
    Args:
        input_files: List of input file paths
        args: Command line arguments
    
    Returns:
        tuple: (batched_tensor, list_of_volumes, list_of_headers, list_of_crop_coords)
    """
    batch_tensors = []
    volumes = []
    headers = []
    crop_coords_list = []
    
    for input_file in input_files:
        # Create temporary args for this input
        temp_args = argparse.Namespace(**vars(args))
        temp_args.input = input_file
        
        # Preprocess individual file
        image_tensor, volume, header, crop_coords = preprocess_input(temp_args)
        
        # Remove the batch dimension (1) from individual tensor to prepare for batching
        image_tensor = image_tensor.squeeze(0)  # Shape: (1, H, W, D)
        
        batch_tensors.append(image_tensor)
        volumes.append(volume)
        headers.append(header)
        crop_coords_list.append(crop_coords)
    
    # Stack tensors along batch dimension
    batched_tensor = Tensor.stack(*batch_tensors, dim=0)  # Shape: (BS, 1, H, W, D)
    
    return batched_tensor, volumes, headers, crop_coords_list


def run_inference(model, image):
    """
    Execute model inference on the preprocessed image.
    
    Args:
        model: The loaded segmentation model
        image: Preprocessed image tensor (single or batched)
        
    Returns:
        Tensor: Raw model output channels
    """
    return model(image)


def run_batch_inference(model, batched_image):
    """
    Execute model inference on batched preprocessed images.
    
    Args:
        model: The loaded segmentation model
        batched_image: Batched preprocessed image tensor (BS, 1, H, W, D)
        
    Returns:
        Tensor: Raw batched model output channels
    """
    return model(batched_image)


def postprocess_output(output_channels, header, crop_coords=None):
    """
    Handle output postprocessing: argmax, padding, and labeling.
    
    Args:
        output_channels: Raw model output tensor
        header: Original NIfTI header
        crop_coords: Coordinates for uncropping (if cropping was applied)
        
    Returns:
        tuple: (processed_labels_data, new_header)
    """
    # Convert model output to segmentation labels
    output = (
        output_channels.argmax(axis=1)
        .rearrange("1 x y z -> z y x")
        .numpy()
        .astype(np.uint8)
    )
    
    # Restore original size if cropping was applied
    if crop_coords is not None:
        output = pad_to_original_size(output, crop_coords)
    
    # Generate labeled output with proper header
    labels, new_header = bwlabel(header, output)
    processed_data = set_header_intent_label(new_header) + labels.tobytes()
    
    return processed_data, new_header


def postprocess_batch_output(batched_output_channels, headers, crop_coords_list):
    """
    Handle batch output postprocessing: argmax, padding, and labeling for multiple outputs.
    
    Args:
        batched_output_channels: Raw batched model output tensor (BS, C, H, W, D)
        headers: List of original NIfTI headers
        crop_coords_list: List of coordinates for uncropping (if cropping was applied)
        
    Returns:
        list: List of (processed_labels_data, new_header) tuples
    """
    results = []
    batch_size = batched_output_channels.shape[0]
    
    for i in range(batch_size):
        # Extract individual output from batch
        output_channels = batched_output_channels[i:i+1]  # Keep batch dimension for consistency
        header = headers[i]
        crop_coords = crop_coords_list[i]
        
        # Process individual output
        processed_data, new_header = postprocess_output(output_channels, header, crop_coords)
        results.append((processed_data, new_header))
    
    return results


def write_output(processed_data, args):
    """
    Handle file output operations including niimath commands and subprocess calls.
    
    Args:
        processed_data: Processed segmentation data ready for output
        args: Command line arguments containing output settings
    """
    output_dtype = "char"
    
    # Handle class probability export if requested
    if args.export_classes:
        # Note: This requires access to output_channels, will need to be called separately
        print(f"brainchop :: Exported classes to c[channel_number]_{args.output}")
    
    # Determine gzip compression based on file extension
    gzip_flag = "0" if str(args.output).endswith(".nii") else "1"
    
    # Build base niimath command
    cmd = ["niimath", "-"]
    if args.inverse_conform and args.model != "mindgrab":
        cmd += ["-reslice_nn", args.input]
    
    # Handle mindgrab-specific processing
    data_to_write = processed_data
    if args.model == "mindgrab":
        cmd = ["niimath", str(args.input)]
        
        # Apply border growth if specified
        if args.border > 0:
            data_to_write = grow_border(processed_data, args.border)
        
        # Write mask file if requested
        if args.mask is not None:
            cmdm = ["niimath", "-"]
            cmdm += ["-reslice_nn", args.input]
            subprocess.run(
                cmdm + ["-gz", "1", args.mask, "-odt", "char"],
                input=data_to_write,
                check=True,
            )
        
        cmd += ["-reslice_mask", "-"]
        output_dtype = "input_force"
    
    # Finalize command and execute
    cmd += ["-gz", gzip_flag, str(args.output), "-odt", output_dtype]
    subprocess.run(cmd, input=data_to_write, check=True)


def run_cli():
    """Main CLI function that orchestrates brainchop command-line operations."""
    parser = get_parser()
    args = parser.parse_args()

    # Handle simple commands that don't require processing
    if args.update:
        update_models()
        return
    if args.list:
        list_models()
        return
    if not args.input:
        parser.print_help()
        return

    # Prepare file paths - convert input list to absolute paths
    input_files = [os.path.abspath(input_file) for input_file in args.input]
    
    # Store original output value before converting to absolute path
    original_output = args.output
    args.output = os.path.abspath(args.output)
    
    print(f"brainchop :: Processing {len(input_files)} input file(s)")

    # Determine model name
    modelname = args.model
    if args.skull_strip:
        modelname = "mindgrab"
        args.model = modelname

    # Check if this is the first run for this model/batch_size combination
    batch_size = args.batch_size
    if not args.no_optimize and is_first_run(modelname, batch_size):
        # Prompt for optimization on first run
        optimization_success = prompt_for_optimization(modelname, batch_size)
        if optimization_success:
            print(f"brainchop :: Model optimized successfully. Continuing with processing...")
        print()  # Add blank line for clarity

    # Check for cached optimization and set BEAM environment variable
    best_beam = get_best_beam_for_batch_size(modelname, batch_size)
    original_beam = os.environ.get("BEAM")
    
    if best_beam is not None:
        os.environ["BEAM"] = str(best_beam)
        print(f"brainchop :: Using cached optimization BEAM={best_beam} for batch size {batch_size}")
    
    # Load model (will use the BEAM environment variable if set)
    model = get_model(modelname)
    print(f"brainchop :: Loaded model {modelname}")

    # Process input files in batches
    print(f"brainchop :: Using batch size: {batch_size}")
    
    for batch_start in range(0, len(input_files), batch_size):
        batch_end = min(batch_start + batch_size, len(input_files))
        batch_files = input_files[batch_start:batch_end]
        
        # Process batch using proper batching
        batched_tensor, volumes, headers, crop_coords_list = preprocess_batch(batch_files, args)
        
        batched_output_channels = run_batch_inference(model, batched_tensor)
        
        batch_results = postprocess_batch_output(batched_output_channels, headers, crop_coords_list)
        
        # Process each file's results
        for i, input_file in enumerate(batch_files):
            global_index = batch_start + i + 1
            processed_data, new_header = batch_results[i]
            
            # Generate output filename based on input filename, model, and index
            # Always use the new naming format unless user explicitly specified a custom output
            if original_output == "output.nii.gz":
                # Default output - always use the new dynamic naming format
                output_file = generate_output_filename(input_file, modelname, global_index, None)
            elif len(input_files) == 1:
                # Single file with custom output specified - use the custom output
                output_file = args.output
            else:
                # Multiple files with custom output directory - generate dynamic name in that directory
                output_dir = str(Path(args.output).parent)
                output_file = generate_output_filename(input_file, modelname, global_index, output_dir)
            
            print(f"Processing file {global_index}/{len(input_files)}: {input_file} -> {output_file}")
            
            # Create a temporary args object for this specific input
            current_args = argparse.Namespace(**vars(args))
            current_args.input = input_file
            current_args.output = output_file
            
            # Handle class export before writing main output
            if args.export_classes:
                # For batched processing, we need to extract the individual output channels
                individual_output_channels = batched_output_channels[i:i+1]
                export_classes(individual_output_channels, headers[i], current_args.output)
                print(f"brainchop :: Exported classes to c[channel_number]_{current_args.output}")
            
            write_output(processed_data, current_args)
    
    # Save optimization data to cache if BEAM was used (and not already saved during pre-optimization)
    current_beam = os.environ.get("BEAM")
    if current_beam is not None and not is_first_run(modelname, batch_size):
        try:
            beam_value = int(current_beam)
            save_optimization_cache(modelname, batch_size, beam_value)
        except ValueError:
            pass  # Invalid BEAM value, skip caching
    
    # Restore original BEAM environment variable
    if original_beam is not None:
        os.environ["BEAM"] = original_beam
    elif "BEAM" in os.environ:
        del os.environ["BEAM"]
    
    cleanup()


if __name__ == "__main__":
    run_cli()
```

can you help me finish this implementation? the save to nifti and nanchecks
aren't actually implemented in this file.

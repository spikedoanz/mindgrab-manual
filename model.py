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
        print(self.weights)
        
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

    def _welford_pass1_kernel(self, C, D, H, W):
        """Pass 1: per-block Welford (stable) -> store mean_block and M2_block"""
        K = 16  # Block size
        GX, GY, GZ = W // K, H // K, D // K
        return f"""
        @group(0) @binding(0) var<storage, read> data1: array<f32>;      // IN
        @group(0) @binding(1) var<storage, read_write> data2: array<f32>; // SCRATCH

        const W: u32 = {W}u; const H: u32 = {H}u; const D: u32 = {D}u;
        const CIN: u32 = {C}u;
        const K: u32 = {K}u; 
        const GX: u32 = {GX}u; const GY: u32 = {GY}u; const GZ: u32 = {GZ}u;
        const GRID: u32 = GX * GY * GZ;

        const MEAN_GRID_BASE: u32 = 0u;
        const M2_GRID_BASE: u32 = MEAN_GRID_BASE + CIN * GRID;

        fn idx_in(c:u32, x:u32, y:u32, z:u32) -> u32 {{ return c*(D*H*W) + (z*H + y)*W + x; }}
        fn idx_grid(c:u32, bx:u32, by:u32, bz:u32) -> u32 {{ return c*GRID + (bz*GY + by)*GX + bx; }}

        @compute @workgroup_size(1, 1, 1)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
            let bx = gid.x; let by = gid.y; let cz = gid.z;
            if (bx >= GX || by >= GY || cz >= GZ * CIN) {{ return; }}

            let c: u32 = cz / GZ;
            let bz: u32 = cz % GZ;
            if (c >= CIN) {{ return; }}

            let x0 = bx * K; let y0 = by * K; let z0 = bz * K;
            var n: f32 = 0.0;
            var mean: f32 = 0.0;
            var M2: f32 = 0.0;

            for (var dz: u32 = 0u; dz < K; dz++) {{
                let z = z0 + dz;
                for (var dy: u32 = 0u; dy < K; dy++) {{
                    let y = y0 + dy;
                    for (var dx: u32 = 0u; dx < K; dx++) {{
                        let x = x0 + dx;
                        let v: f32 = data1[idx_in(c, x, y, z)];
                        let n1 = n + 1.0;
                        let delta = v - mean;
                        let mean1 = mean + delta / n1;
                        let delta2 = v - mean1;
                        M2 = M2 + delta * delta2;
                        n = n1;
                        mean = mean1;
                    }}
                }}
            }}
            let gi = idx_grid(c, bx, by, bz);
            data2[MEAN_GRID_BASE + gi] = mean;
            data2[M2_GRID_BASE + gi] = M2;
        }}
        """

    def _welford_pass2_kernel(self, C, D, H, W):
        """Pass 2: reduce blocks -> global μ[c], invσ[c] via Chan combination"""
        K = 16
        GX, GY, GZ = W // K, H // K, D // K
        GRID = GX * GY * GZ
        return f"""
        @group(0) @binding(0) var<storage, read_write> data2: array<f32>; // SCRATCH

        const W: u32 = {W}u; const H: u32 = {H}u; const D: u32 = {D}u;
        const CIN: u32 = {C}u;
        const K: u32 = {K}u;
        const GX: u32 = {GX}u; const GY: u32 = {GY}u; const GZ: u32 = {GZ}u;
        const GRID: u32 = {GRID}u;
        const EPS: f32 = 1e-5;

        const MEAN_GRID_BASE: u32 = 0u;
        const M2_GRID_BASE: u32 = MEAN_GRID_BASE + CIN * GRID;
        const MU_BASE: u32 = M2_GRID_BASE + CIN * GRID;
        const INVSTD_BASE: u32 = MU_BASE + CIN;

        fn combine(nA:f32, mA:f32, M2A:f32, nB:f32, mB:f32, M2B:f32) -> vec3<f32> {{
            let n = nA + nB;
            if (n == 0.0) {{ return vec3<f32>(0.0, 0.0, 0.0); }}
            let delta = mB - mA;
            let mean = mA + delta * (nB / n);
            let M2 = M2A + M2B + delta * delta * (nA * nB / n);
            return vec3<f32>(n, mean, M2);
        }}

        @compute @workgroup_size(1, 1, 1)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
            let c = gid.x;
            if (c >= CIN) {{ return; }}

            var n: f32 = 0.0;
            var mean: f32 = 0.0;
            var M2: f32 = 0.0;
            let baseMean = MEAN_GRID_BASE + c * GRID;
            let baseM2 = M2_GRID_BASE + c * GRID;
            let nBlock: f32 = f32(K * K * K);

            for (var i: u32 = 0u; i < GRID; i++) {{
                let mB = data2[baseMean + i];
                let M2B = data2[baseM2 + i];
                let comb = combine(n, mean, M2, nBlock, mB, M2B);
                n = comb.x;
                mean = comb.y;
                M2 = comb.z;
            }}

            let var_val = max(0.0, M2 / max(n, 1.0));
            let inv = inverseSqrt(var_val + EPS);
            data2[MU_BASE + c] = mean;
            data2[INVSTD_BASE + c] = inv;
        }}
        """

    def _welford_pass3_kernel(self, C, D, H, W, apply_gelu):
        """Pass 3: apply z-score (+ optional QuickGELU) safely in f32"""
        K = 16
        GX, GY, GZ = W // K, H // K, D // K
        GRID = GX * GY * GZ
        apply_gelu_str = "true" if apply_gelu else "false"
        return f"""
        @group(0) @binding(0) var<storage, read_write> data0: array<f32>; // OUT
        @group(0) @binding(1) var<storage, read> data1: array<f32>;      // IN
        @group(0) @binding(2) var<storage, read> data2: array<f32>;      // SCRATCH

        const W: u32 = {W}u; const H: u32 = {H}u; const D: u32 = {D}u;
        const CIN: u32 = {C}u;
        const K: u32 = {K}u;
        const GRID: u32 = {GRID}u;

        const MEAN_GRID_BASE: u32 = 0u;
        const M2_GRID_BASE: u32 = MEAN_GRID_BASE + CIN * GRID;
        const MU_BASE: u32 = M2_GRID_BASE + CIN * GRID;
        const INVSTD_BASE: u32 = MU_BASE + CIN;
        const APPLY_GELU: bool = {apply_gelu_str};

        fn idx_in(c:u32, x:u32, y:u32, z:u32) -> u32 {{ return c*(D*H*W) + (z*H + y)*W + x; }}
        fn idx_out(c:u32, x:u32, y:u32, z:u32) -> u32 {{ return c*(D*H*W) + (z*H + y)*W + x; }}

        fn sigmoid_f(x: f32) -> f32 {{
            return 1.0 / (1.0 + exp(-x));
        }}

        @compute @workgroup_size(8, 8, 2)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
            let x = gid.x; let y = gid.y; let z = gid.z;
            if (x >= W || y >= H || z >= D) {{ return; }}

            for (var c: u32 = 0u; c < CIN; c++) {{
                let mu = data2[MU_BASE + c];
                let inv = data2[INVSTD_BASE + c];
                let v = data1[idx_in(c, x, y, z)];
                let zn = (v - mu) * inv;

                // --- SOLUTION: Add safety checks for NaN and Infinity ---
                // A common way to check for NaN is (zn != zn).
                // We also check for large values that might indicate Infinity.
                if (zn != zn || abs(zn) > 1e10) {{
                    data0[idx_out(c, x, y, z)] = 0.0;
                }} else {{
                    let yv = select(zn, zn * sigmoid_f(1.702 * zn), APPLY_GELU);
                    // Final check in case the GELU approximation creates another NaN
                    if (yv != yv) {{
                        data0[idx_out(c, x, y, z)] = 0.0;
                    }} else {{
                        data0[idx_out(c, x, y, z)] = yv;
                    }}
                }}
            }}
        }}
        """

    def _apply_welford_groupnorm_gelu(self, input_bytes, shape, apply_gelu):
        """Orchestrates the 3-pass Welford-based GroupNorm and optional GELU."""
        C, D, H, W = shape
        K = 16
        GX, GY, GZ = W // K, H // K, D // K
        GRID = GX * GY * GZ

        # Calculate scratch buffer size
        mean_grid_size = C * GRID
        m2_grid_size = C * GRID
        mu_size = C
        invstd_size = C
        scratch_elements = mean_grid_size + m2_grid_size + mu_size + invstd_size
        scratch_size_bytes = scratch_elements * 4  # f32
        
        # Align all buffer sizes to 16 bytes
        input_size_bytes = len(input_bytes)
        output_size_bytes = input_size_bytes
        input_aligned_size = ((input_size_bytes + 15) // 16) * 16
        output_aligned_size = ((output_size_bytes + 15) // 16) * 16
        scratch_aligned_size = ((scratch_size_bytes + 15) // 16) * 16
        
        # Pad input bytes if necessary
        padded_input_bytes = input_bytes
        if input_aligned_size > input_size_bytes:
            padded_input_bytes += b'\x00' * (input_aligned_size - input_size_bytes)

        # Create buffers
        input_buffer = utils.create_buffer(self.dev, input_aligned_size, webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopyDst)
        output_buffer = utils.create_buffer(self.dev, output_aligned_size, webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopySrc)
        scratch_buffer = utils.create_buffer(self.dev, scratch_aligned_size, webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopyDst | webgpu.WGPUBufferUsage_CopySrc)
        utils.write_buffer(self.dev, input_buffer, 0, bytearray(padded_input_bytes))

        # --- Pass 1 ---
        kernel1_src = self._welford_pass1_kernel(C, D, H, W)
        shader_module1 = utils.create_shader_module(self.dev, kernel1_src)
        bindings1 = [
            {"binding": 0, "resource": {"buffer": input_buffer, "offset": 0, "size": input_aligned_size}},
            {"binding": 1, "resource": {"buffer": scratch_buffer, "offset": 0, "size": scratch_aligned_size}}
        ]
        bind_group_layout1 = utils.create_bind_group_layout(self.dev, entries=[
            {"binding": 0, "visibility": webgpu.WGPUShaderStage_Compute, "buffer": {"type": webgpu.WGPUBufferBindingType_ReadOnlyStorage}},
            {"binding": 1, "visibility": webgpu.WGPUShaderStage_Compute, "buffer": {"type": webgpu.WGPUBufferBindingType_Storage}}
        ])
        pipeline1 = utils.create_compute_pipeline(self.dev, layout=utils.create_pipeline_layout(self.dev, [bind_group_layout1]), compute={"module": shader_module1, "entry_point": "main"})
        bind_group1 = utils.create_bind_group(self.dev, layout=bind_group_layout1, entries=bindings1)
        
        # --- Pass 2 ---
        kernel2_src = self._welford_pass2_kernel(C, D, H, W)
        shader_module2 = utils.create_shader_module(self.dev, kernel2_src)
        bindings2 = [{"binding": 0, "resource": {"buffer": scratch_buffer, "offset": 0, "size": scratch_aligned_size}}]
        bind_group_layout2 = utils.create_bind_group_layout(self.dev, entries=[
            {"binding": 0, "visibility": webgpu.WGPUShaderStage_Compute, "buffer": {"type": webgpu.WGPUBufferBindingType_Storage}}
        ])
        pipeline2 = utils.create_compute_pipeline(self.dev, layout=utils.create_pipeline_layout(self.dev, [bind_group_layout2]), compute={"module": shader_module2, "entry_point": "main"})
        bind_group2 = utils.create_bind_group(self.dev, layout=bind_group_layout2, entries=bindings2)

        # --- Pass 3 ---
        kernel3_src = self._welford_pass3_kernel(C, D, H, W, apply_gelu)
        shader_module3 = utils.create_shader_module(self.dev, kernel3_src)
        bindings3 = [
            {"binding": 0, "resource": {"buffer": output_buffer, "offset": 0, "size": output_aligned_size}},
            {"binding": 1, "resource": {"buffer": input_buffer, "offset": 0, "size": input_aligned_size}},
            {"binding": 2, "resource": {"buffer": scratch_buffer, "offset": 0, "size": scratch_aligned_size}}
        ]
        bind_group_layout3 = utils.create_bind_group_layout(self.dev, entries=[
            {"binding": 0, "visibility": webgpu.WGPUShaderStage_Compute, "buffer": {"type": webgpu.WGPUBufferBindingType_Storage}},
            {"binding": 1, "visibility": webgpu.WGPUShaderStage_Compute, "buffer": {"type": webgpu.WGPUBufferBindingType_ReadOnlyStorage}},
            {"binding": 2, "visibility": webgpu.WGPUShaderStage_Compute, "buffer": {"type": webgpu.WGPUBufferBindingType_ReadOnlyStorage}}
        ])
        pipeline3 = utils.create_compute_pipeline(self.dev, layout=utils.create_pipeline_layout(self.dev, [bind_group_layout3]), compute={"module": shader_module3, "entry_point": "main"})
        bind_group3 = utils.create_bind_group(self.dev, layout=bind_group_layout3, entries=bindings3)

        # --- Command Encoding and Submission ---
        encoder = utils.create_command_encoder(self.dev)
        # Pass 1 Dispatch
        pass1 = utils.begin_compute_pass(encoder)
        utils.set_pipeline(pass1, pipeline1)
        utils.set_bind_group(pass1, bind_group1)
        utils.dispatch_workgroups(pass1, GX, GY, GZ * C)
        utils.end_compute_pass(pass1)
        # Pass 2 Dispatch
        pass2 = utils.begin_compute_pass(encoder)
        utils.set_pipeline(pass2, pipeline2)
        utils.set_bind_group(pass2, bind_group2)
        utils.dispatch_workgroups(pass2, C, 1, 1)
        utils.end_compute_pass(pass2)
        # Pass 3 Dispatch
        pass3 = utils.begin_compute_pass(encoder)
        utils.set_pipeline(pass3, pipeline3)
        utils.set_bind_group(pass3, bind_group3)
        utils.dispatch_workgroups(pass3, W // 8, H // 8, D // 2)
        utils.end_compute_pass(pass3)
        
        cb = utils.command_encoder_finish(encoder)
        utils.submit(self.dev, [cb])
        utils.sync(self.dev)
        
        result_bytes = utils.read_buffer(self.dev, output_buffer)
        return result_bytes[:output_size_bytes]

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

            # === NEW: Integrated GroupNorm + GELU ===
            if i < len(self.config['layers']) - 1:
                use_bnorm = self.config.get('bnorm', True)
                use_gelu = self.config.get('gelu', True)

                if use_bnorm:
                    print(f"Applying Welford GroupNorm (GELU: {use_gelu})")
                    current_bytes = self._apply_welford_groupnorm_gelu(
                        input_bytes=current_bytes,
                        shape=current_shape,
                        apply_gelu=use_gelu
                    )
                    
                    # Save intermediate output after the combined operation
                    output_name_tag = "groupnorm_gelu" if use_gelu else "groupnorm"
                    norm_gelu_output = np.frombuffer(current_bytes, dtype=np.float32).reshape(current_shape)
                    for c in range(current_shape[0]):
                        save_array_to_nifti(norm_gelu_output[c], f"layer_{i+1:02d}_{output_name_tag}_c{c}.nii.gz")
                    print(f"Saved {current_shape[0]} {output_name_tag} output channels to layer_{i+1:02d}_{output_name_tag}_c*.nii.gz")

        result = np.frombuffer(current_bytes, dtype=np.float32).reshape(current_shape)
        return result


# Example usage
if __name__ == "__main__":
    from brainchop.niimath import conform
    
    volume, header = conform("conformed.nii.gz")
    input_tensor = volume.astype(np.float32)
    input_tensor = volume.transpose((2, 1, 0)).astype(np.float32)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    print(f"Input shape: {input_tensor.shape}")
    
    model = WebGPUMeshNet(
        "weights/tissue_fast/model.json", 
        "weights/tissue_fast/model.pth"
    )
    
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

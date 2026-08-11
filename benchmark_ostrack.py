import os
import sys
import time
import torch
import numpy as np

# Path configuration
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'viat', 'tracking', 'ostrack_repo'))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from lib.config.ostrack.config import cfg, update_config_from_file
from lib.models.ostrack import build_ostrack
import tensorrt as trt

def benchmark():
    print("=" * 70)
    print("      OSTRACK PERFORMANCE BENCHMARK (PyTorch FP32 vs FP16 vs TensorRT)      ")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"TensorRT Version: {trt.__version__}")
    print("-" * 70)

    # 1. Load PyTorch model
    yaml_file = os.path.join(repo_dir, 'experiments', 'ostrack', 'vitb_256_mae_ce_32x4_ep300.yaml')
    update_config_from_file(yaml_file)

    net = build_ostrack(cfg, training=False)
    ckpt_path = os.path.abspath('viat/checkpoints/OSTrack_ep0300.pth.tar')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    net.load_state_dict(checkpoint['net'], strict=True)
    net.cuda().eval()

    z_dummy = torch.randn(1, 3, 128, 128, device='cuda', dtype=torch.float32)
    x_dummy = torch.randn(1, 3, 256, 256, device='cuda', dtype=torch.float32)
    mask_dummy = torch.zeros((1, 64), device='cuda', dtype=torch.bool)

    # ----------------------------------------------------
    # Benchmark 1: PyTorch FP32
    # ----------------------------------------------------
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Warmup
    with torch.no_grad():
        for _ in range(20):
            _ = net(template=z_dummy, search=x_dummy, ce_template_mask=mask_dummy)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    
    iters = 100
    start_evt.record()
    with torch.no_grad():
        for _ in range(iters):
            _ = net(template=z_dummy, search=x_dummy, ce_template_mask=mask_dummy)
    end_evt.record()
    torch.cuda.synchronize()

    fp32_ms = start_evt.elapsed_time(end_evt) / float(iters)
    fp32_vram_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
    fp32_vram_res = torch.cuda.memory_reserved() / (1024 * 1024)

    # ----------------------------------------------------
    # Benchmark 2: PyTorch FP16 (AMP Autocast)
    # ----------------------------------------------------
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Warmup
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        for _ in range(20):
            _ = net(template=z_dummy, search=x_dummy, ce_template_mask=mask_dummy)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        for _ in range(iters):
            _ = net(template=z_dummy, search=x_dummy, ce_template_mask=mask_dummy)
    end_evt.record()
    torch.cuda.synchronize()

    fp16_ms = start_evt.elapsed_time(end_evt) / float(iters)
    fp16_vram_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
    fp16_vram_res = torch.cuda.memory_reserved() / (1024 * 1024)

    # Clean up PyTorch model memory
    del net
    torch.cuda.empty_cache()

    # ----------------------------------------------------
    # Benchmark 3: Native TensorRT 11 Engine (.engine)
    # ----------------------------------------------------
    engine_path = os.path.abspath('checkpoints/ostrack.engine')
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"TensorRT engine not found at {engine_path}")

    trt_logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    score_map = torch.empty(1, 1, 16, 16, device='cuda', dtype=torch.float32)
    size_map = torch.empty(1, 2, 16, 16, device='cuda', dtype=torch.float32)
    offset_map = torch.empty(1, 2, 16, 16, device='cuda', dtype=torch.float32)

    context.set_tensor_address('template', z_dummy.data_ptr())
    context.set_tensor_address('search', x_dummy.data_ptr())
    context.set_tensor_address('score_map', score_map.data_ptr())
    context.set_tensor_address('size_map', size_map.data_ptr())
    context.set_tensor_address('offset_map', offset_map.data_ptr())

    stream = torch.cuda.Stream()
    stream_handle = stream.cuda_stream

    # Warmup
    for _ in range(20):
        context.execute_async_v3(stream_handle=stream_handle)
    stream.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record(stream=stream)
    for _ in range(iters):
        context.execute_async_v3(stream_handle=stream_handle)
    end_evt.record(stream=stream)
    stream.synchronize()

    trt_ms = start_evt.elapsed_time(end_evt) / float(iters)
    trt_vram_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
    trt_vram_res = torch.cuda.memory_reserved() / (1024 * 1024)

    # ----------------------------------------------------
    # Results Summary Table
    # ----------------------------------------------------
    print(f"{'Model Version':<25} | {'Latency (ms)':<14} | {'FPS':<10} | {'VRAM Alloc (MB)':<16} | {'VRAM Reserved (MB)':<18}")
    print("-" * 92)
    print(f"{'PyTorch FP32':<25} | {fp32_ms:<14.2f} | {1000.0/fp32_ms:<10.1f} | {fp32_vram_alloc:<16.2f} | {fp32_vram_res:<18.2f}")
    print(f"{'PyTorch FP16 (AMP)':<25} | {fp16_ms:<14.2f} | {1000.0/fp16_ms:<10.1f} | {fp16_vram_alloc:<16.2f} | {fp16_vram_res:<18.2f}")
    print(f"{'TensorRT 11 Engine (.engine)':<25} | {trt_ms:<14.2f} | {1000.0/trt_ms:<10.1f} | {trt_vram_alloc:<16.2f} | {trt_vram_res:<18.2f}")
    print("=" * 92)

if __name__ == '__main__':
    benchmark()

import os
import sys
import argparse
import importlib
import torch

repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'viat', 'tracking', 'ostrack_repo'))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from lib.config.ostrack.config import cfg, update_config_from_file
from lib.models.ostrack import build_ostrack

def export_onnx():
    yaml_file = os.path.join(repo_dir, 'experiments', 'ostrack', 'vitb_256_mae_ce_32x4_ep300.yaml')
    update_config_from_file(yaml_file)

    net = build_ostrack(cfg, training=False)
    
    ckpt_candidates = [
        os.path.abspath('viat/checkpoints/OSTrack_ep0300.pth.tar'),
        os.path.abspath('checkpoints/OSTrack_ep0300.pth.tar')
    ]
    ckpt_path = None
    for c in ckpt_candidates:
        if os.path.exists(c):
            ckpt_path = c
            break
            
    if not ckpt_path:
        raise FileNotFoundError("OSTrack checkpoint not found in candidates.")

    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    net.load_state_dict(checkpoint['net'], strict=True)
    net.eval()
    net.export_onnx = True

    dummy_z = torch.randn(1, 3, 128, 128)
    dummy_x = torch.randn(1, 3, 256, 256)

    output_onnx = os.path.abspath('checkpoints/ostrack.onnx')
    os.makedirs(os.path.dirname(output_onnx), exist_ok=True)

    print(f"Exporting ONNX to {output_onnx}...")
    torch.onnx.export(
        net,
        (dummy_z, dummy_x),
        output_onnx,
        verbose=False,
        opset_version=17,
        input_names=["template", "search"],
        output_names=["score_map", "size_map", "offset_map"],
        dynamo=False
    )
    print(f"Successfully exported ONNX to {output_onnx}! Size: {os.path.getsize(output_onnx) / (1024*1024):.2f} MB")

    try:
        import onnx
        import onnxsim
        print("Simplifying ONNX graph with onnxsim...")
        model_sim, flag = onnxsim.simplify(output_onnx)
        if flag:
            onnx.save(model_sim, output_onnx)
            print(f"Simplified ONNX graph successfully! Final size: {os.path.getsize(output_onnx) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"Skipping onnxsim simplification: {e}")

if __name__ == '__main__':
    export_onnx()

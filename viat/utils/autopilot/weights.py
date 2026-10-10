"""Local-only weight lookup for Autopilot. Nothing here ever downloads.

Hugging Face models are looked up, in order, at
  <repo>/checkpoints/<org>/<name>, <repo>/checkpoints/<name>,
  <repo>/viat/checkpoints/zero_shot/<org>_<name>, and the local Hugging Face cache.
Ultralytics weights (.pt) must sit in <repo>/checkpoints/.
"""
import glob
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CHECKPOINTS = os.path.join(PROJECT_ROOT, 'checkpoints')

# model id -> (what it is used for, approximate download size)
KNOWN_WEIGHTS = {
    'IDEA-Research/grounding-dino-base': ('Grounding DINO base detector', '~0.9 GB'),
    'IDEA-Research/grounding-dino-tiny': ('Grounding DINO tiny detector', '~0.7 GB'),
    'google/owlv2-base-patch16-ensemble': ('OWLv2 base detector', '~0.6 GB'),
    'google/owlv2-large-patch14-ensemble': ('OWLv2 large detector', '~1.7 GB'),
    'openai/clip-vit-base-patch32': ('CLIP judge (installed)', '~0.6 GB'),
    'openai/clip-vit-large-patch14': ('CLIP-L judge (more accurate)', '~1.7 GB'),
    'Qwen/Qwen2.5-VL-7B-Instruct': ('vision-language judge', '~16 GB'),
    'Qwen/Qwen3-VL-8B-Instruct': ('vision-language judge', '~17 GB'),
    'sam3.1_l.pt': ('SAM 3 text detector and zoom-in judge', '~3.4 GB'),
}


def _valid_dir(path):
    return os.path.isdir(path) and (os.path.exists(os.path.join(path, 'config.json')) or
                                    os.path.exists(os.path.join(path, 'preprocessor_config.json')))


def resolve_hf(model_id):
    """Return a local directory holding ``model_id`` or None (never downloads)."""
    if os.path.isabs(model_id):
        return model_id if _valid_dir(model_id) else None
    org, _, name = model_id.partition('/')
    name = name or org
    candidates = [os.path.join(CHECKPOINTS, model_id), os.path.join(CHECKPOINTS, name),
                  os.path.join(PROJECT_ROOT, 'viat', 'checkpoints', 'zero_shot', model_id.replace('/', '_'))]
    hf_home = os.environ.get('HF_HOME') or os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
    candidates += sorted(glob.glob(os.path.join(hf_home, 'hub', f"models--{model_id.replace('/', '--')}",
                                                'snapshots', '*')))
    for c in candidates:
        if _valid_dir(c):
            return c
    return None


def resolve_file(name):
    path = name if os.path.isabs(name) else os.path.join(CHECKPOINTS, name)
    return path if os.path.isfile(path) else None


def is_available(key):
    if key.endswith('.pt') or key.endswith('.engine'):
        return resolve_file(key) is not None
    return resolve_hf(key) is not None


def missing_files(model_id):
    """Small files a transformers model folder lacks (the weights alone are not enough)."""
    folder = os.path.join(CHECKPOINTS, model_id)
    if not os.path.isdir(folder):
        return None
    have = set(os.listdir(folder))
    need = ['config.json', 'preprocessor_config.json', 'tokenizer_config.json']
    if not have & {'tokenizer.json', 'vocab.txt', 'vocab.json'}:
        need.append('tokenizer.json (or vocab files)')
    if not any(f.endswith('.safetensors') or f.endswith('.bin') for f in have):
        need.append('model.safetensors')
    return [f for f in need if f not in have]


def missing_message(key):
    lacking = None if key.endswith(('.pt', '.engine')) else missing_files(key)
    if lacking:
        return (f"{os.path.join(CHECKPOINTS, key)} is incomplete, also needs: {', '.join(lacking)} "
                f"(the small files from the same Hugging Face repo)")
    what, size = KNOWN_WEIGHTS.get(key, ('model', 'unknown size'))
    where = (os.path.join(CHECKPOINTS, key) if key.endswith('.pt')
             else os.path.join(CHECKPOINTS, key) + '/  (config.json + weights)')
    return f"weights needed: {key} ({what}, {size}) -> put them in {where}"

"""Portable SIR run layout; preserves checkpoint directory names."""
import json
from pathlib import Path

def model_dir(exp_dir): return Path(exp_dir)/'model'
def ensure_result_layout(exp_dir):
    m,e=model_dir(exp_dir),Path(exp_dir)/'eval'
    m.mkdir(parents=True,exist_ok=True); e.mkdir(parents=True,exist_ok=True)
    return m,e
def load_run_config_doc(exp_dir):
    p=Path(exp_dir)/'run_config.json'
    return json.loads(p.read_text()) if p.is_file() else {}
def resolve_eval_seed(exp_dir,cli_seed=None):
    doc=load_run_config_doc(exp_dir) if exp_dir else {}
    return int(doc.get('seed',cli_seed if cli_seed is not None else 2025))

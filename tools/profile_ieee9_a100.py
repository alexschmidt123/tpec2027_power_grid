"""Timing-only IEEE 9-bus evaluation; never trains policies."""
import argparse, copy, hashlib, json, platform, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--repo', required=True)
p.add_argument('--T', type=int, required=True, choices=[3,4,5])
p.add_argument('--output', required=True)
p.add_argument('--systems', type=int, default=512)
p.add_argument('--eval-seed', type=int, default=1001)
p.add_argument('--warmup-seed', type=int, default=91001)
a = p.parse_args()
if a.systems < 2 or a.eval_seed == a.warmup_seed:
    p.error('Use at least two test episodes and a separate warm-up seed')
root = Path(a.repo).resolve()
sys.path.insert(0, str(root))
import numpy as np
import torch
from src.config import load_config
from src.hardware import hardware_info
from src.objectives.eig import continuous_eig as c
from src.domains.swing.continuous_rocof import EndpointRocofObserver
if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
    raise RuntimeError('This measurement requires an NVIDIA A100 GPU')
rel = ('09142026/09142026_ieee9_eig_endpoint_rocof_T3_train101_eval1001/T3/run' if a.T == 3 else f'09152026/09152026_endpoint_rocof_T4-5_train101_eval1001_rerun12h/T{a.T}/run')
source = root / 'TPEC_conference/results/experiments' / rel
metadata = json.loads((source/'run_config.json').read_text())
args = argparse.Namespace(**metadata['settings'])
assert args.seed == 101 and args.T == a.T and args.contrasts == 128
assert metadata['observation_kind'] == 'endpoint_rocof'
assert json.loads((source/'completion.json').read_text())['records'] > 0
cfgpath = source/'source_snapshot/configs'/Path(args.config).name
cfg = load_config(str(cfgpath))
cfg.raw = copy.deepcopy(metadata['physical_config'])
observer = EndpointRocofObserver(cfg, duration_bounds=(args.duration_min,args.duration_max), injection_bus=args.bus, amplitude=args.amplitude, window=args.window)
engine = c.OnlineEIG(cfg, observer, horizon=args.T, sigma=args.noise_sigma, contrasts=args.contrasts, min_separation=args.min_duration_separation)
policies, checkpoints = {}, {}
for method in ['fixed','dad','rl_sboed']:
    path = source/'models'/f'{method}.pth'
    blob = torch.load(path, map_location='cpu', weights_only=True)
    assert blob['method'] == method and blob['horizon'] == args.T
    assert blob['n_obs'] == engine.n_obs and blob['protocol'] == metadata['protocol']
    assert blob['settings']['seed'] == 101
    policy = c.REDQPolicy(args.T, engine.n_obs) if method == 'rl_sboed' else c.DurationPolicy(args.T, engine.n_obs, fixed=method=='fixed')
    policy.load_state_dict(blob['state_dict'], strict=True)
    policy.eval()
    policies[method] = policy
    checkpoints[method] = hashlib.sha256(path.read_bytes()).hexdigest()
# Final manuscript uses four Step-DAD updates; original training archive
# settings may describe an earlier sixteen-update evaluation. No retraining.
args.refinement_updates = 4
args.methods = 'random,fixed,dad,rl_sboed,myopic,step_dad'
args.eval_seed = a.warmup_seed
args.eval_systems = 1
# All methods receive one discarded warm-up history using the original budgets.
c.evaluate(engine, policies, args)
torch.cuda.synchronize()
args.eval_seed = a.eval_seed
args.eval_systems = a.systems
out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)
manifest = {'purpose':'timing_only_no_retraining', 'evaluation_override':{'refinement_updates':{'training_archive':metadata['settings']['refinement_updates'],'paper':4}}, 'network':'ieee9','T':args.T,'training_seed':101,'evaluation_seed':args.eval_seed,'systems_per_method':a.systems,'warmup_histories_per_method':1,'warmup_evaluation_seed':a.warmup_seed,'hardware':hardware_info(),'checkpoint_sha256':checkpoints,'source_run':str(source),'settings':vars(args),'source_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (root/'src').rglob('*.py')},'timing_definition':'sum of original evaluator decision_seconds_per_stage, including posterior updates and refinement; excludes true-system simulation and EIG scoring'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
rows = c.evaluate(engine, policies, args, record_path=out/'rollouts.jsonl')
results = []
for method in args.methods.split(','):
    values = [sum(r['decision_seconds_per_stage']) for r in rows if r['method']==method]
    assert len(values)==a.systems and np.all(np.isfinite(values))
    results.append({'network':'ieee9','T':args.T,'method':method,'phase':'online','seed':101,'seconds':float(np.mean(values)),'sd_seconds':float(np.std(values,ddof=1)),'systems':a.systems,'hardware':manifest['hardware'],'source_manifest':str(out/'manifest.json')})
(out/'timing.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2),flush=True)

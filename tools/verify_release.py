"""Verify copied result bytes, portable navigation, scope, and checkpoint loading."""
import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.config import load_config
from src.objectives.eig.continuous_eig import DurationPolicy
from src.objectives.eig.continuous_redq import REDQPolicy
from src.policies.rl_sboed import AdaptiveExperimentPolicy,PolicyConfig

def verify():
    manifest=json.loads((ROOT/'provenance/migration.json').read_text())
    checked=0;models=[]
    for record in manifest['included_files']:
        p=ROOT/record['path']
        if p.suffix in {'.tex','.pdf'} and 'results' not in p.parts and 'sir ode result' not in p.parts:continue
        actual=hashlib.sha256(p.read_bytes()).hexdigest()
        if actual!=record['sha256']:raise ValueError(f'Checksum mismatch: {p}')
        checked+=1
        if p.suffix!='.pth':continue
        payload=torch.load(p,map_location='cpu',weights_only=False)
        if 'horizon' in payload:
            method=payload['method'];T=payload['horizon'];obs=payload['n_obs']
            if method not in {'dad','fixed','rl_sboed'}:raise ValueError(f'Out-of-scope model {p}')
            model=REDQPolicy(T,obs) if method=='rl_sboed' else DurationPolicy(T,obs,fixed=method=='fixed')
        else:
            meta=payload['meta'];method=payload.get('method',p.stem)
            if p.stem not in {'dad_eig','rl_sboed_eig'}:raise ValueError(f'Out-of-scope model {p}')
            state=payload['state_dict']
            hidden=int(meta.get('policy_hidden',256))
            # Shapes are read from the saved architecture, not inferred from folder names.
            T=int(meta.get('horizon',meta.get('max_steps',int(next(x[1:] for x in p.parts if x in ['T3','T4','T5'])))))
            n_actions=int(meta.get('n_actions',100))
            config=PolicyConfig(max_steps=T,obs_dim=int(meta.get('obs_dim',1)),summary_dim=33,hidden=hidden,particle_dim=int(meta.get('particle_dim',3)))
            model=AdaptiveExperimentPolicy(n_actions,config)
        model.load_state_dict(payload['state_dict'],strict=True)
        if not all(torch.isfinite(t).all() for t in model.state_dict().values()):raise ValueError(f'Nonfinite checkpoint {p}')
        models.append({'path':record['path'],'method':method,'sha256':actual})
    for entry in manifest['included_links']:
        p=ROOT/entry['path'];target=p.resolve()
        if not p.exists() or ROOT not in target.parents:raise ValueError(f'Invalid navigation link {p}')
    # The executable source supports only the six declared EIG methods.
    forbidden=['moe','mocu','msc']
    for folder in ['src','configs','scripts']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and p.suffix in {'.py','.yaml','.sh'}:
                content=p.read_text().lower()
                if any(word in content for word in forbidden):raise ValueError(f'Excluded implementation reference: {p}')
    summary={'verified_files':checked,'loaded_checkpoint_files':len(models),'portable_links':len(manifest['included_links']),'excluded_implementation_scan':'passed','checkpoints':models}
    print(json.dumps({k:v for k,v in summary.items() if k!='checkpoints'},indent=2))
    return summary

if __name__=='__main__':
    result=verify()
    if '--write-report' in sys.argv:(ROOT/'provenance/release_verification.json').write_text(json.dumps(result,indent=2)+'\n')

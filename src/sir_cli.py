"""Fresh six-method SIR benchmark using the archived experiment's implementations."""
import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from src.config import ALL_METHODS,load_config,resolve_config_path,repo_root
from src.domains.sir.banks import generate_sir_bank
from src.domains.sir.context import build_sir_context
from src.objectives.eig.vector import train_vector_eig_policy,evaluate_vector_eig
from src.hardware import hardware_info

METHOD_MAP=dict(dad='dad_eig',rl_sboed='rl_sboed_eig',step_dad='step_dad',myopic='myopic_delta_h',fixed='fixed_open_loop',random='random')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--T',type=int,default=3)
    p.add_argument('--seed',type=int,default=101)
    p.add_argument('--eval-seeds',default='1001,1002,1003,1004,1005')
    p.add_argument('--method','--methods',default=','.join(ALL_METHODS))
    p.add_argument('--objective',choices=['eig'],default='eig')
    p.add_argument('--experiment_type','--experiment-type',choices=['eig_based'],default='eig_based')
    p.add_argument('--noise_sigma','--noise-sigma',type=float,default=None)
    p.add_argument('--N_obs',type=int,default=1,help='Ignored for scalar SIR observations')
    p.add_argument('--output','--exp-dir',default=None)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    methods=args.method.split(',');seeds=[int(v) for v in args.eval_seeds.split(',')]
    if not methods or len(set(methods))!=len(methods) or not set(methods)<=set(ALL_METHODS):p.error('Select distinct supported methods')
    if args.T<1 or not seeds or len(set(seeds))!=len(seeds):p.error('Positive T and distinct evaluation seeds required')
    cfg=load_config(resolve_config_path(args.config))
    if cfg.topology not in ['sir','sir_ode']:p.error('SIR configuration required')
    cfg.raw['experiment'].update(step_number=args.T,methods=methods)
    cfg.raw.pop('step_number',None)
    cfg.raw.setdefault('observation',{})['N_obs']=1
    if args.noise_sigma is not None:cfg.raw['observation']['noise_sigma']=args.noise_sigma
    now=datetime.now();out=Path(args.output) if args.output else repo_root()/'experiments'/now.strftime('%m%d%Y')/now.strftime(f'%m%d%Y_%H%M%S_%f_sir_eig_T{args.T}_train{args.seed}')
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    if args.smoke:
        cfg.raw['data']['dataset_dir']=str(out/'smoke_bank')
    try:
        bank=generate_sir_bank(cfg,smoke=args.smoke,force=False)
        metadata={**cfg.raw,'seed':args.seed,'training_seed':args.seed,'evaluation_seeds':seeds,'hardware':hardware_info(),'settings':vars(args),'data_generation_record':bank,'source_hashes':{}}
        for path in (repo_root()/'src').rglob('*.py'):
            rel=path.relative_to(repo_root());metadata['source_hashes'][str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
            target=out/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        (out/'run_config.json').write_text(json.dumps(metadata,indent=2)+'\n')
        ctx=build_sir_context(cfg,out_dir=out,smoke=args.smoke)
        for method in ['dad','rl_sboed']:
            if method in methods or (method=='dad' and 'step_dad' in methods):
                train_vector_eig_policy(ctx,method=METHOD_MAP[method],smoke=args.smoke,seed=args.seed)
        for seed in seeds:
            evaluate_vector_eig(ctx,smoke=args.smoke,methods=tuple(METHOD_MAP[m] for m in methods),eval_seed=seed)
            dest=out/'evaluations'/f'seed_{seed}';dest.mkdir(parents=True)
            shutil.move(str(out/'eval'),str(dest/'eval'))
            shutil.copy2(out/'run_config.json',dest/'run_config.json')
        (out/'evaluation_manifest.json').write_text(json.dumps({'training_seed':args.seed,'evaluation_seeds':seeds,'model_scope':'local_to_this_run_folder'},indent=2)+'\n')
        (out/'completion.json').write_text(json.dumps({'objective':'eig','methods':methods,'evaluation_seeds':seeds,'wall_seconds':time.monotonic()-start,'smoke_only':args.smoke},indent=2)+'\n')
        (out/'exit_code').write_text('0\n')
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps({'exception':type(exc).__name__,'message':str(exc)}))
        (out/'exit_code').write_text('1\n');raise
    print('SIR_EIG_COMPLETE',out,flush=True)

if __name__=='__main__':main()

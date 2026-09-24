"""Dispatch fresh conference EIG experiments and explicit Cartesian sweeps."""
import argparse
import itertools
import subprocess
import sys
from pathlib import Path
from src.config import load_config,resolve_config_path
ROOT=Path(__file__).resolve().parents[1]

def main():
    argv=sys.argv[1:]
    if '--sweep' not in argv:
        p=argparse.ArgumentParser(add_help=False)
        p.add_argument('--config',default='configs/ieee9_eig.yaml')
        known,_=p.parse_known_args(argv)
        cfg=load_config(resolve_config_path(known.config))
        script='sir_run.sh' if cfg.topology.startswith('sir') else 'continuous_eig.sh'
        raise SystemExit(subprocess.call(['bash',str(ROOT/'scripts'/script),*argv],cwd=ROOT))
    argv.remove('--sweep')
    p=argparse.ArgumentParser(description='Fresh training per configuration/T/seed cell.')
    p.add_argument('--configs','--config',default='ieee9_eig')
    p.add_argument('--T',default='3')
    p.add_argument('--seed',default='101')
    p.add_argument('--eval-seeds',default='1001,1002,1003')
    args,other=p.parse_known_args(argv)
    if any(v.split('=')[0] in ['--output','--evaluate-from','--exp-dir'] for v in other):p.error('Sweeps allocate fresh outputs; per-run reuse is unsupported')
    for config,T,seed in itertools.product(args.configs.split(','),args.T.split(','),args.seed.split(',')):
        subprocess.run(['bash',str(ROOT/'run.sh'),'--config',config,'--T',T,'--seed',seed,'--eval-seeds',args.eval_seeds,*other],cwd=ROOT,check=True)
if __name__=='__main__':main()

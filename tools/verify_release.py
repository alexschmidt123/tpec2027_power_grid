"""Verify public file integrity, portable links, and optional checkpoint loading."""
import argparse,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TOP={'src','configs','scripts','tools','provenance','TPEC_conference','README.md','requirements.txt','run.sh','sweep_run.sh','.gitignore','.gitattributes'}
MANIFEST=ROOT/'provenance/public_release_manifest.json'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def public_paths():
 for name in sorted(TOP):
  start=ROOT/name
  if not start.exists():continue
  paths=[start] if start.is_file() else start.rglob('*')
  for p in paths:
   if '__pycache__' in p.parts or 'build' in p.relative_to(ROOT).parts:continue
   if p==MANIFEST or p.is_symlink() or not p.is_file():continue
   yield p
def create_manifest():
 files=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest(p)} for p in public_paths()]
 links=[{'path':str(p.relative_to(ROOT)),'target':str(p.readlink())} for name in TOP if (ROOT/name).is_dir() for p in (ROOT/name).rglob('*') if p.is_symlink()]
 MANIFEST.write_text(json.dumps({'scope':'Current public working tree; private archives and ignored experiments excluded.','files':files,'links':links},indent=2)+'\n')
def verify(checkpoints=False):
 manifest=json.loads(MANIFEST.read_text()); errors=[]
 for r in manifest['files']:
  p=ROOT/r['path']
  if not p.is_file() or digest(p)!=r['sha256']:errors.append('Checksum mismatch: '+r['path'])
 known={x['path'] for x in manifest['files']}
 for p in public_paths():
  if str(p.relative_to(ROOT)) not in known:errors.append('Unlisted public file: '+str(p.relative_to(ROOT)))
 for r in manifest['links']:
  p=ROOT/r['path']
  if not p.is_symlink() or str(p.readlink())!=r['target'] or not p.exists() or not p.resolve().is_relative_to(ROOT):errors.append('Invalid symlink: '+r['path'])
 links=0
 for p in public_paths():
  if p.suffix=='.md':
   for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
    if target.startswith(('https://','http://','mailto:','#')):continue
    target=target.split('#')[0]
    if target and not (p.parent/target).exists():errors.append('Broken Markdown link: '+str(p.relative_to(ROOT))+' -> '+target)
    links+=1
  if p.suffix in {'.json','.jsonl','.md','.txt','.yaml','.sh'}:
   text=p.read_text()
   if re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}',text):errors.append('Possible credential: '+str(p.relative_to(ROOT)))
   if re.search(r'/home/(?:grads|qiqi)|/scratch/user|/Users/gaoming',text):errors.append('Machine-specific path: '+str(p.relative_to(ROOT)))
 loaded=0
 if checkpoints:
  import sys,torch
  sys.path.insert(0,str(ROOT))
  from src.objectives.eig.continuous_eig import DurationPolicy
  from src.objectives.eig.continuous_redq import REDQPolicy
  for p in public_paths():
   if p.suffix!='.pth':continue
   payload=torch.load(p,map_location='cpu',weights_only=False)
   method=payload['method'];T=payload['horizon'];obs=payload['n_obs']
   model=REDQPolicy(T,obs) if method=='rl_sboed' else DurationPolicy(T,obs,fixed=method=='fixed')
   model.load_state_dict(payload['state_dict'],strict=True)
   assert all(torch.isfinite(t).all() for t in model.state_dict().values()),str(p)
   loaded+=1
 if errors:raise SystemExit('\n'.join(errors))
 print(json.dumps({'verified_files':len(manifest['files']),'portable_symlinks':len(manifest['links']),'markdown_links_checked':links,'loaded_checkpoints':loaded,'credential_and_path_scan':'passed'},indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--write-manifest',action='store_true');parser.add_argument('--checkpoints',action='store_true');args=parser.parse_args()
 if args.write_manifest:create_manifest()
 verify(args.checkpoints)

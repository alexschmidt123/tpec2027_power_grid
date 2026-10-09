import json,math,statistics,collections,hashlib
from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description='Validate and aggregate archived IEEE 14-bus rescoring results.')
parser.add_argument('--archive-root',type=Path,default=Path(__file__).resolve().parents[1]/'results/larger_L')
base=parser.parse_args().archive_root; rows=[json.loads(x) for x in (base/'scores.jsonl').read_text().splitlines()]; assert len(rows)==64512
keys=set();groups=collections.defaultdict(list);errors=[0.,0.]
for r in rows:
 key=(r['T'],r['method'],r['evaluation_seed'],r['system'])+(() if r['method'] in ['random','myopic'] else (r['training_seed'],));assert key not in keys;keys.add(key)
 assert r['evaluation_seed'] in [1001,1002,1003] and r['training_seed'] in [101,202,303] and 0<=r['system']<512
 assert sorted(x['L'] for x in r['scores'])==[10000,100000]
 errors[0]=max(errors[0],r['terminal_state_max_abs_error']);errors[1]=max(errors[1],r['archived_score_absolute_error'])
 for score in [{'L':128,'spce_nats':r['original_spce_nats']},*r['scores']]:
  assert math.isfinite(score['spce_nats']) and score['spce_nats']<=math.log(score['L']+1)+1e-10
  seed=r['evaluation_seed'] if r['method'] in ['random','myopic'] else r['training_seed']
  groups[r['T'],r['method'],score['L'],seed].append(score)
summary=[]
for T in [3,4,5]:
 for method in ['random','fixed','myopic','dad','rl_sboed','step_dad']:
  seeds=[1001,1002,1003] if method in ['random','myopic'] else [101,202,303]
  for L in [128,10000,100000]:
   cells=[groups[T,method,L,seed] for seed in seeds];assert all(len(c)==(512 if method in ['random','myopic'] else 1536) for c in cells)
   means=[statistics.mean(x['spce_nats'] for x in c) for c in cells];item={'T':T,'method':method,'L':L,'mean':statistics.mean(means),'sd_of_seed_means':statistics.stdev(means),'seed_means':dict(zip(seeds,means)),'histories':sum(map(len,cells))}
   if L>128:
    allscores=[x for c in cells for x in c];item['mean_snmc']=statistics.mean(x['snmc_nats'] for x in allscores);item['mean_sample_bound_gap']=statistics.mean(x['bound_gap_nats'] for x in allscores);item['near_ceiling_fraction_0_01_nat']=sum(math.log(L+1)-x['spce_nats']<.01 for x in allscores)/len(allscores)
   summary.append(item)
report={'histories':len(rows),'unique_identities':len(keys),'max_terminal_state_error':errors[0],'max_archived_L128_score_error':errors[1],'scores_sha256':hashlib.sha256((base/'scores.jsonl').read_bytes()).hexdigest(),'scorer_sha256':hashlib.sha256((Path(__file__).resolve().parent/'archived_ieee14_rescorer.py').read_bytes()).hexdigest(),'rows':summary}
assert report['scorer_sha256']==json.loads((base/'manifest.json').read_text())['scorer_sha256']
(base/'publication_summary.json').write_text(json.dumps(report,indent=2)+'\n')
print({k:v for k,v in report.items() if k!='rows'})
for T in [3,4,5]:
 print('T',T)
 for method in ['random','fixed','myopic','dad','rl_sboed','step_dad']:
  print(method,[(r['L'],round(r['mean'],4),round(r['sd_of_seed_means'],4),round(r.get('mean_sample_bound_gap',0),4),round(r.get('near_ceiling_fraction_0_01_nat',0),4)) for r in summary if r['T']==T and r['method']==method])

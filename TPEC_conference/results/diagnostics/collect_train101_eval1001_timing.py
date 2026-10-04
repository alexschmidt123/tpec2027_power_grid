import json,pathlib,statistics,hashlib
import argparse
parser=argparse.ArgumentParser(description='Recalculate paper costs from train101/eval1001 archives')
parser.add_argument('--results-root', type=pathlib.Path, required=True)
parser.add_argument('--output', type=pathlib.Path, required=True)
args=parser.parse_args()
root=args.results_root
off=json.loads((root/'diagnostics/offline_timing_sources.json').read_text()); output={'training_seed':101,'evaluation_seed':1001,'online_definition':'mean over 512 episodes of sum(decision_seconds_per_stage); no warm-up exclusion added retrospectively','rows':[]}
for network in ['ieee9','ieee14']:
 source_rows=json.loads((root/network/'01_results/online_times_by_run.json').read_text())
 for T in [3,4,5]:
  stages={}
  for method in ['fixed','dad','rl_sboed']:
   source=next(x for x in off[network] if x['T']==T and x['method']==method)['sources'][0]
   f=root/source; conf=json.loads((f/'run_config.json').read_text()); assert conf['settings']['seed']==101
   summary=json.loads((f/'summary.json').read_text()); r=next(x for x in summary if x['method']==method)
   stages[method]={'seconds':r['training_seconds'],'source':str(f.relative_to(root)), 'hardware':conf['hardware']}
  for method in ['random','fixed','dad','rl_sboed','myopic','step_dad']:
   src=next(x for x in source_rows if x['T']==T and x['method']==method and x['training_seed']==101)
   f=root/src['source_run']; conf=json.loads((f/'run_config.json').read_text()); assert conf['settings']['seed']==101
   if method=='step_dad': assert conf['settings']['refinement_updates']==4
   raw=f/'rollouts.json'; rows=json.loads(raw.read_text()); selected=[r for r in rows if r['method']==method and r['evaluation_seed']==1001]; assert len(selected)==512,(network,T,method,len(selected))
   times=[sum(r['decision_seconds_per_stage']) for r in selected]; assert all(len(r['decision_seconds_per_stage'])==T for r in selected)
   offline_seconds=0
   if method=='fixed': offline_seconds=stages['fixed']['seconds']
   if method=='rl_sboed': offline_seconds=stages['rl_sboed']['seconds']
   if method in ['dad','step_dad']: offline_seconds=stages['fixed']['seconds']+stages['dad']['seconds']
   output['rows'].append({'network':network,'T':T,'method':method,'offline_seconds':offline_seconds,'offline_hours':offline_seconds/3600,'offline_stages':stages if method in ['dad','step_dad'] else ({method:stages[method]} if method in stages else {}),'online_mean_ms':statistics.mean(times)*1000,'online_sd_ms':statistics.stdev(times)*1000,'online_episodes':len(times),'online_source':str(f.relative_to(root)),'online_hardware':conf['hardware'],'rollouts_sha256':hashlib.sha256(raw.read_bytes()).hexdigest()})
args.output.write_text(json.dumps(output,indent=2)+'\n')
for r in output['rows']: print(r['network'],r['T'],r['method'],'offline_h=',round(r['offline_hours'],6),'online_ms=',round(r['online_mean_ms'],6),r['online_hardware']['gpu_model'])

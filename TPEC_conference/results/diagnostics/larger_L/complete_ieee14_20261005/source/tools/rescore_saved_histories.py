"""Evaluation-only sPCE/sNMC scoring of fixed, archived grid histories.

No policies are loaded, trained, refined, or rerun. Contrastive likelihoods
carry each candidate's state through the observed durations. Gaussian
normalizers cancel because all histories share the specified noise model.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time
import types

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def information_bounds(log_true, log_contrast_sum, count):
    """Sample scores; lower/upper bound interpretation holds in expectation."""
    return (float(log_true - np.logaddexp(log_true, log_contrast_sum)
                  + math.log(count + 1)),
            float(log_true - log_contrast_sum + math.log(count)))


def history_likelihood(observer, parameters, durations, observations, sigma):
    state = observer.initial_state(len(parameters))
    result = np.zeros(len(parameters))
    for duration, observation in zip(durations, observations):
        response = observer.propagate(parameters, state, np.full(len(parameters), duration))
        state = response.terminal_state
        result += -0.5 * np.sum(((observation - response.observations) / sigma) ** 2, axis=1)
    return result, state


def score_history(observer, metadata, row, levels, chunk_size, rescore_seed=20260929):
    lower = np.asarray(metadata['prior_lower'])
    upper = np.asarray(metadata['prior_upper'])
    theta = np.asarray(row['true_MK'])
    durations = np.asarray(row['duration_sequence_s'])
    observations = np.asarray(row['observations_rocof_hz_s'])
    sigma = metadata['settings']['noise_sigma']
    if observations.shape != (len(durations), 1) or theta.shape != lower.shape:
        raise ValueError('Inconsistent history dimensions')
    if np.any(theta < lower) or np.any(theta > upper):
        raise ValueError('Generating parameters outside saved prior')
    if len(durations) != metadata['settings']['T']:
        raise ValueError('History horizon does not match saved configuration')
    gap = metadata['settings']['min_duration_separation']
    if np.any(np.diff(durations) < gap - 1e-9):
        raise ValueError('History violates archived duration-order constraints')
    log_true, terminal = history_likelihood(observer, theta[None], durations, observations, sigma)
    terminal_error = float(np.max(np.abs(terminal[0] - row['true_terminal_state'])))
    if terminal_error > 1e-8:
        raise ValueError(f'Simulator does not reproduce archived terminal state: {terminal_error}')
    # Same contrastive stream for matching seed/episode across methods/training seeds.
    rng = np.random.default_rng(np.random.SeedSequence(
        [rescore_seed, int(row['evaluation_seed']), int(row['system'])]))
    contrast_sum = -np.inf
    count = 0
    results = []
    start = time.perf_counter()
    for level in sorted(set(levels)):
        while count < level:
            n = min(chunk_size, level - count)
            contrasts = rng.uniform(lower, upper, size=(n, len(lower)))
            likelihood, _ = history_likelihood(observer, contrasts, durations, observations, sigma)
            contrast_sum = np.logaddexp(contrast_sum, logsumexp(likelihood))
            count += n
        spce, snmc = information_bounds(float(log_true[0]), float(contrast_sum), count)
        results.append({'L': count, 'spce_nats': spce, 'snmc_nats': snmc,
                        'bound_gap_nats': snmc - spce,
                        'cumulative_scoring_seconds': time.perf_counter() - start})
    return {'method': row['method'], 'evaluation_seed': row['evaluation_seed'],
            'system': row['system'], 'T': len(durations),
            'original_spce_nats': row['terminal_spce_nats'],
            'terminal_state_max_abs_error': terminal_error,
            'scores': results}


def reproduce_archived_score(observer, metadata, row):
    """Regenerate the exact archived prior contrasts, before increasing L."""
    settings = metadata['settings']
    rng = np.random.default_rng(row['evaluation_seed'])
    block = settings['contrasts'] + 1
    target = int(row['system'])
    for _ in range(target + 1):
        parameters = rng.uniform(metadata['prior_lower'], metadata['prior_upper'],
                                 size=(block, len(metadata['prior_lower'])))
    if not np.array_equal(parameters[0], row['true_MK']):
        raise ValueError('Archived parameter random stream differs from expected sampling')
    likelihood, _ = history_likelihood(observer, parameters,
        row['duration_sequence_s'], np.asarray(row['observations_rocof_hz_s']), settings['noise_sigma'])
    score = float(likelihood[0] - logsumexp(likelihood) + math.log(block))
    error = abs(score - row['terminal_spce_nats'])
    if error > 1e-8:
        raise ValueError(f'Archived sPCE score mismatch: {error}')
    return error


def make_observer(metadata, simulator_root):
    from src.config import SBOEDConfig
    from src.domains.swing.continuous_rocof import EndpointRocofObserver
    for relative in ['src/domains/swing/continuous.py', 'src/domains/swing/continuous_cuda.py',
                     'src/domains/swing/continuous_rocof.py', 'src/domains/swing/cuda.py',
                     'src/domains/swing/simulator.py', 'src/config.py',
                     'src/domains/swing/design.py', 'src/control/cuda_control.py']:
        expected = metadata['source_hashes'].get(relative)
        if hashlib.sha256((simulator_root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Archived simulator hash mismatch: ' + relative)
    if metadata['observation_kind'] != 'endpoint_rocof':
        raise ValueError('This scorer supports archived signed endpoint RoCoF only')
    cfg = SBOEDConfig(metadata['physical_config'], config_path=simulator_root / 'configs' / Path(metadata['settings']['config']).name)
    settings = metadata['settings']
    observer = EndpointRocofObserver(cfg,
        duration_bounds=(settings['duration_min'], settings['duration_max']),
        injection_bus=settings['bus'], amplitude=settings['amplitude'], window=settings['window'])
    if abs(observer.rocof_sample_dt - metadata['rocof_sample_dt']) > 1e-12:
        raise ValueError('Observation differencing interval changed')
    return observer


def result_key(row):
    key = (row['T'], row['method'], row['evaluation_seed'], row['system'])
    return key if row['method'] in ('random', 'myopic') else key + (row['training_seed'],)


def load_completed_results(path, levels):
    if not path.exists():
        return []
    results = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = set()
    for row in results:
        key = result_key(row)
        if key in keys:
            raise ValueError('Duplicate completed result: ' + str(key))
        if sorted(score['L'] for score in row['scores']) != sorted(set(levels)):
            raise ValueError('Completed result contrast levels differ')
        if not all(np.isfinite(score[field]) for score in row['scores']
                   for field in ('spce_nats', 'snmc_nats')):
            raise ValueError('Nonfinite completed score')
        keys.add(key)
    return results


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulator-root', type=Path, default=ROOT,
                        help='Exact archived source snapshot root for simulator hash verification')
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--levels', default='128,10000,100000')
    parser.add_argument('--chunk-size', type=int, default=10000)
    parser.add_argument('--pilot-per-method', type=int, default=2,
                        help='0 scores all archived histories; positive values select a timing pilot')
    parser.add_argument('--resume', action='store_true', help='Resume only a matching manifest and valid complete records')
    parser.add_argument('--rescore-seed', type=int, default=20260929)
    args = parser.parse_args()
    levels = [int(v) for v in args.levels.split(',')]
    if min(levels) < 1 or args.chunk_size < 1 or args.pilot_per_method < 0:
        parser.error('Positive contrast levels/chunks and nonnegative pilot count required')
    sys.path.insert(0, str(args.simulator_root.resolve()))
    # Archived snapshots omit the top-level empty __init__.py. Bind the package
    # explicitly so imports cannot silently fall back to the current source.
    package = types.ModuleType("src")
    package.__path__ = [str(args.simulator_root.resolve() / "src")]
    sys.modules["src"] = package
    manifest = {'runs': [{
        'path':str(run.resolve()),
        'run_config_sha256':hashlib.sha256((run/'run_config.json').read_bytes()).hexdigest(),
        'rollouts_sha256':hashlib.sha256((run/'rollouts.json').read_bytes()).hexdigest()}
        for run in args.runs], 'levels':sorted(set(levels)),
        'chunk_size':args.chunk_size, 'rescore_seed':args.rescore_seed,
        'pilot_per_method':args.pilot_per_method, 'simulator_root':str(args.simulator_root.resolve()),
        'scorer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.resume:
        if json.loads((args.output/'manifest.json').read_text()) != manifest:
            raise ValueError('Resume manifest mismatch: settings, code, or inputs changed')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        atomic_json(args.output/'manifest.json', manifest)
    all_results = load_completed_results(args.output/'scores.jsonl', levels)
    completed_keys = {result_key(row) for row in all_results}
    completed_at_start = len(all_results)
    print(json.dumps({'resuming':args.resume, 'already_completed':completed_at_start}), flush=True)
    inventory = []
    seen = {}
    overall_start = time.perf_counter()
    for run in args.runs:
        metadata = json.loads((run / 'run_config.json').read_text())
        records = json.loads((run / 'rollouts.json').read_text())
        observer = make_observer(metadata, args.simulator_root)
        selected = []
        per_method = {}
        for row in records:
            if args.pilot_per_method and per_method.get(row['method'], 0) >= args.pilot_per_method:
                continue
            identity = (metadata['settings']['T'], row['method'], row['evaluation_seed'], row['system'])
            # Random/Myopic have no training-seed dependence in this archived protocol.
            key = identity if row['method'] in ('random', 'myopic') else identity + (metadata['settings']['seed'],)
            signature = tuple(json.dumps(row[k], sort_keys=True) for k in
                ('true_MK', 'duration_sequence_s', 'observations_rocof_hz_s', 'terminal_spce_nats'))
            if key in seen:
                if seen[key] != signature:
                    raise ValueError('Duplicate history identity contains different data: ' + str(key))
                continue
            seen[key] = signature
            selected.append(row)
            per_method[row['method']] = per_method.get(row['method'], 0) + 1
        inventory.append({'run': str(run), 'records': len(records), 'selected': len(selected),
            'run_config_sha256': hashlib.sha256((run/'run_config.json').read_bytes()).hexdigest(),
            'rollouts_sha256': hashlib.sha256((run/'rollouts.json').read_bytes()).hexdigest()})
        for row in selected:
            key = (metadata['settings']['T'], row['method'], row['evaluation_seed'], row['system'])
            if row['method'] not in ('random', 'myopic'):
                key += (metadata['settings']['seed'],)
            if key in completed_keys:
                continue
            history_start = time.perf_counter()
            check = reproduce_archived_score(observer, metadata, row)
            result = score_history(observer, metadata, row, levels, args.chunk_size, args.rescore_seed)
            result.update(training_seed=metadata['settings']['seed'], source_run=str(run),
                          total_history_wall_seconds=time.perf_counter()-history_start,
                          archived_score_absolute_error=check)
            with (args.output/'scores.jsonl').open('a') as stream:
                stream.write(json.dumps(result)+'\n')
            all_results.append(result)
            completed_keys.add(key)
            if len(all_results) % 10 == 0 or len(all_results) == 1:
                atomic_json(args.output/'progress.json', {
                    'status':'running', 'histories_completed':len(all_results),
                    'histories_added_this_session':len(all_results)-completed_at_start,
                    'session_elapsed_seconds':time.perf_counter()-overall_start,
                    'last_history':list(key), 'updated_unix':time.time()})
            print(json.dumps({'T':result['T'], 'method':result['method'], 'system':result['system'],
                'L':result['scores'][-1]['L'], 'seconds':result['scores'][-1]['cumulative_scoring_seconds'],
                'spce':result['scores'][-1]['spce_nats'], 'snmc':result['scores'][-1]['snmc_nats']}), flush=True)
    if completed_keys != set(seen):
        raise ValueError('Completed history identities do not match selected archive')
    summary = {'pilot_only': bool(args.pilot_per_method), 'levels': levels,
        'chunk_size':args.chunk_size, 'rescore_seed':args.rescore_seed,
        'simulator_root':str(args.simulator_root),
        'scorer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'histories_scored':len(all_results), 'wall_seconds':time.perf_counter()-overall_start,
        'inventory':inventory, 'bound_interpretation':'Bounds hold in expectation; empirical gaps are not EIG confidence intervals.',
        'by_T':{}}
    for horizon in sorted(set(row['T'] for row in all_results)):
        group = [row for row in all_results if row['T']==horizon]
        summary['by_T'][horizon] = {str(level):{
            'mean_seconds_per_history':float(np.mean([next(s['cumulative_scoring_seconds'] for s in r['scores'] if s['L']==level) for r in group])),
            'min_seconds_per_history':float(min(next(s['cumulative_scoring_seconds'] for s in r['scores'] if s['L']==level) for r in group)),
            'max_seconds_per_history':float(max(next(s['cumulative_scoring_seconds'] for s in r['scores'] if s['L']==level) for r in group))}
            for level in levels}
    atomic_json(args.output/'summary.json', summary)
    atomic_json(args.output/'progress.json', {
        'status':'complete', 'histories_completed':len(all_results),
        'session_elapsed_seconds':time.perf_counter()-overall_start, 'updated_unix':time.time()})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

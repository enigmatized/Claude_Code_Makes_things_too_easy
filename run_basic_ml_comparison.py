"""Save matched training-only downside/upside × pooled/per-stock experiments.

Run explicitly with python3 run_basic_ml_comparison.py. Viewing the app never
calls this script. Historical outputs and the final submission pin are preserved.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from research import RunConfig, default_model_registry, default_registry
from research.legacy import PROJECT_ROOT
from research.store import ExperimentStore


def run_comparison(store, config, output, local_models=('logistic',), progress=print):
    if config.evaluation != 'training_validation' or not config.include_ml:
        raise ValueError('Use a training-validation ML configuration.')
    if not set(local_models).issubset(config.models) or not local_models:
        raise ValueError('Local comparison models must be a nonempty subset of pooled models.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plan = {'created_utc': datetime.now(timezone.utc).isoformat(), 'base_config': asdict(config),
            'per_stock_models': list(local_models), 'targets': ['loss', 'upside'],
            'scopes': ['pooled', 'per_stock'],
            'protocol': 'Fixed chronological training validation; same portfolio rules and factors; no test reads, threshold search or tuning.'}
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    records, predictions, portfolios = [], [], []
    reference = None
    for target in plan['targets']:
        for scope in plan['scopes']:
            requested = replace(config, prediction_target=target, model_scope=scope,
                                models=local_models if scope == 'per_stock' else config.models)
            started = time.monotonic()
            result = store.get_or_run(requested, progress=progress)
            originals = result.monthly.loc[result.monthly.variant.eq('baseline'),
                                          ['strategy', 'return_month', 'simple_return']].reset_index(drop=True)
            if reference is None:
                reference = originals
            else:
                pd.testing.assert_frame_equal(originals, reference)
            pred_rows = []
            for mid, batch in result.prediction_batches.items():
                sources = batch.predictions.get('prediction_source', pd.Series('pooled', index=batch.predictions.index)).value_counts()
                pred_rows.append({'model_id':mid, 'prediction_rows':len(batch.predictions),
                                  'fit_summary_rows':len(batch.fits), 'prediction_sources':sources.to_dict()})
            run_id = result.cache_info['run_id']
            size = sum(p.stat().st_size for p in (store.root/run_id).rglob('*') if p.is_file())
            record = {'target':target,'scope':scope,'run_id':run_id,'models':list(requested.models),
                      'source':result.cache_info['source'],'elapsed_seconds':time.monotonic()-started,
                      'artifact_bytes':size,'predictions':pred_rows}
            records.append(record)
            predictions.append(result.model_metrics.assign(target=target, scope=scope, run_id=run_id))
            portfolios.append(result.metrics.assign(target=target, scope=scope, run_id=run_id))
            progress(json.dumps(record))
    pd.concat(predictions, ignore_index=True).to_csv(output/'predictive_comparison.csv',index=False)
    pd.concat(portfolios, ignore_index=True).to_csv(output/'portfolio_comparison.csv',index=False)
    record = {**plan, 'runs':records,'identical_original_return_histories':True,
              'total_artifact_bytes':sum(r['artifact_bytes'] for r in records)}
    (output/'comparison.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--artifact-root',type=Path)
    parser.add_argument('--cash-momentum-months',type=int,default=3)
    args = parser.parse_args()
    output = args.output or PROJECT_ROOT/'0_6_Basic_ML_Directions/outputs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    config = RunConfig(tuple(default_registry()), include_ml=True, models=tuple(default_model_registry()),
                       cash_momentum_months=args.cash_momentum_months)
    record = run_comparison(ExperimentStore(args.artifact_root),config,output)
    print(f"Saved matched comparison: {output}; {record['total_artifact_bytes']/1024**2:.1f} MiB",flush=True)


if __name__ == '__main__':
    main()

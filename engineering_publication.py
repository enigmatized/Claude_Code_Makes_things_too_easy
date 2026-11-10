"""Add completed feature-engineering evidence to the hosted release.

Only aggregate curves/metrics, fold/search tables and feature-construction audits
are published. Model objects, prediction panels and extended CSVs stay local.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid

import pandas as pd

from research.store import digest_file


def check_engineering(study,window):
    scope=study['scope'];audits=study['audits'];monthly=study['monthly']
    if window not in {'training','test'} or scope['window']!=window:raise ValueError('Wrong engineering window.')
    if bool(scope['test_csv_loaded'])!=(window=='test'):raise ValueError('Wrong source data scope.')
    if pd.Timestamp(scope['trained_through_return_month'])>pd.Timestamp(scope['fit_origin']):raise ValueError('Future fit labels.')
    dates=pd.to_datetime(monthly.return_month)
    if dates.empty or dates.min()!=pd.Timestamp(scope['first_return_month']) or dates.max()!=pd.Timestamp(scope['last_return_month']):raise ValueError('Wrong engineering dates.')
    if window=='training' and dates.max()>pd.Timestamp('2009-12-31'):raise ValueError('Test data in training tab.')
    if window=='test' and (dates.min()<pd.Timestamp('2010-01-31') or dates.max()>pd.Timestamp('2018-12-31')):
        raise ValueError('Official-test dates must be within 2010–2018.')
    if pd.Timestamp(scope['fit_origin'])>pd.to_datetime(monthly.snapshot_date).min():raise ValueError('Fit occurs after an evaluation decision.')
    if not (pd.to_datetime(monthly.snapshot_date)+pd.offsets.MonthEnd(1)).equals(dates):raise ValueError('Wrong return earning month.')
    if not audits.n_features.between(1,10).all():raise ValueError('Model input limit exceeded.')
    for value in audits.audit_json:
        audit=json.loads(value)
        if (pd.Timestamp(audit['fit_last_snapshot'])>=pd.Timestamp(audit['fit_origin'])
                or pd.Timestamp(audit['fit_origin'])!=pd.Timestamp(scope['fit_origin'])):raise ValueError('Feature fit crossed its origin.')


def export_engineering(source_publication,study_directory,destination):
    from published_results import PublishedStrategies
    from research.engineering_study import load_engineering_study,FRAMES
    source=PublishedStrategies(source_publication);destination=Path(destination).resolve();study_directory=Path(study_directory)
    if destination.exists():raise FileExistsError('Choose a new immutable publication directory.')
    loaded={window:load_engineering_study(study_directory,window) for window in ('training','test')}
    for window,study in loaded.items():check_engineering(study,window)
    temporary=destination.parent/('.publishing_'+uuid.uuid4().hex)
    try:
        for name,digest in source.metadata['file_sha256'].items():
            if digest_file(source.root/name)!=digest:raise ValueError(f'Damaged source release: {name}')
        shutil.copytree(source.root,temporary)
        meta=json.loads(json.dumps(source.metadata));section={'study_id':study_directory.name,'metadata':loaded['training']['metadata'],'windows':{}}
        for window,study in loaded.items():
            frames={}
            for name in FRAMES:
                relative=f'engineering/{window}/{name}.parquet';path=temporary/relative;path.parent.mkdir(parents=True,exist_ok=True)
                study[name].to_parquet(path,index=False,compression='zstd')
                meta['file_sha256'][relative]=digest_file(path);frames[name]=relative
            section['windows'][window]={'scope':study['scope'],'frames':frames}
        section['source_study_sha256']=digest_file(study_directory/'study.json')
        meta['tabs']['engineering']=section
        meta['published_tabs']=list(meta['tabs'])
        meta['parent_manifest_sha256']=source.token[1] if isinstance(source.token,tuple) else digest_file(source.root/'manifest.json')
        meta['created_utc']=datetime.now(timezone.utc).isoformat()
        (temporary/'manifest.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary,ignore_errors=True);raise
    return destination


class PublishedEngineering:
    def __init__(self,publication):
        self.publication=publication;self.section=publication.metadata['tabs']['engineering'];self.token=publication.token

    def has_window(self,window):return window in self.section['windows']

    def study(self,window='training'):
        section=self.section['windows'][window]
        study={'metadata':self.section['metadata'],'scope':section['scope'],
               **{name:self.publication.frame(path) for name,path in section['frames'].items()}}
        check_engineering(study,window)
        return study

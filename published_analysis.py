"""Self-contained feature, tuning and final-result publication sections.

Only completed results are exported. No selector, estimator or portfolio is
fitted/replayed. Final chart windows are prepared once from saved monthly returns.
"""
from pathlib import Path
import json

import pandas as pd

from research.store import digest_file


def check_months(monthly, first, last, lower, upper):
    dates = pd.to_datetime(monthly.return_month)
    if (dates.empty or dates.min() < pd.Timestamp(lower) or dates.max() > pd.Timestamp(upper)
            or dates.min() != pd.Timestamp(first) or dates.max() != pd.Timestamp(last)):
        raise ValueError("Saved result dates do not match their declared evaluation window.")
    if not (pd.to_datetime(monthly.snapshot_date) + pd.offsets.MonthEnd(1)).equals(dates):
        raise ValueError("Snapshot outcomes must be earned in the following month.")


def check_features(study):
    from published_results import require_training
    meta = study['metadata']
    if meta['config']['evaluation'] != 'training_validation' or meta.get('checks', {}).get('test_csv_loaded') is not False:
        raise ValueError("Feature selection must use saved training-only evidence.")
    for summary in study['summaries'].values():
        require_training(summary['manifest']['config'], summary['manifest'], summary['monthly'], baseline_only=False)
    fits = study['fits']
    if not pd.to_datetime(fits.trained_through_return_month).le(pd.to_datetime(fits.fit_snapshot)).all():
        raise ValueError("A feature mask used outcomes after its fit snapshot.")


def check_feature_books(study):
    meta = study['metadata']
    if meta.get('protocol') != 'basket_feature_books' or meta.get('test_csv_loaded') is not False:
        raise ValueError("Advanced feature comparisons must be training-only.")
    window = meta['window']
    check_months(study['monthly'], window['first_return_month'], window['last_return_month'], '2000-01-01', '2009-12-31')


def check_tuning(study):
    meta = study['metadata']
    if (meta.get('protocol') != 'training_holdout' or meta.get('test_csv_loaded') is not False
            or meta.get('fit_policy') != 'frozen_at_origin'
            or any('test' in name for name in meta.get('source_sha256', {}))):
        raise ValueError("Tuning publication requires the frozen training holdout.")
    check_months(study['monthly'], meta['holdout_first_return_month'], meta['holdout_last_return_month'], '2000-01-01', '2009-12-31')
    if pd.Timestamp(meta['trained_through_return_month']) > pd.Timestamp(meta['origin_snapshot']):
        raise ValueError("Tuning labels extend beyond the fitting origin.")
    if pd.Timestamp(meta['origin_snapshot']) >= pd.Timestamp(meta['holdout_first_return_month']):
        raise ValueError("Tuning must finish before the holdout earns returns.")


def check_final(study):
    meta = study['metadata']
    if (meta.get('protocol') != 'final_four_books' or meta.get('fit_policy') != 'frozen_at_origin'
            or meta.get('test_csv_loaded') is not True):
        raise ValueError("Final publication requires the recorded frozen test study.")
    check_months(study['monthly'], meta['first_return_month'], meta['last_return_month'], '2010-01-01', '2018-12-31')
    origin = pd.Timestamp(meta['fit_snapshot'])
    if pd.Timestamp(meta['trained_through_return_month']) > origin or origin >= pd.Timestamp(meta['first_return_month']):
        raise ValueError("Final model fitting must precede test returns.")
    fits = study['fits']
    if (not pd.to_datetime(fits.fit_snapshot).le(origin).all()
            or not pd.to_datetime(fits.trained_through_return_month).le(origin).all()):
        raise ValueError("Final models were refitted after the freeze date.")


def export_features(store, study_id, book_directory, write):
    from research.feature_studies import load_feature_study, list_feature_studies
    from research.feature_reporting import method_notes
    from research.basket_features import load_feature_book_scores
    study = load_feature_study(store, study_id)
    check_features(study)
    record = next(row for row in list_feature_studies(store) if row['study_id'] == study_id)
    section = {'record': record, 'metadata': study['metadata'], 'method_notes': method_notes(),
        'frames': {name: write(f'features/{name}.parquet', study[name])
                   for name in ('portfolio_comparison', 'predictive_comparison', 'fits', 'calibration')}, 'summaries': {}}
    for recipe, summary in study['summaries'].items():
        frames, values = {}, {}
        for name, value in summary.items():
            if isinstance(value, pd.DataFrame):
                frames[name] = write(f'features/summaries/{recipe}/{name}.parquet', value)
            else:
                values[name] = value
        section['summaries'][recipe] = {'frames': frames, 'values': values}
    if book_directory:
        path = Path(book_directory)
        books = load_feature_book_scores(path)
        check_feature_books(books)
        section['books'] = {'study_id': path.name, 'source_metadata_sha256': digest_file(path / 'study.json'),
            'metadata': books['metadata'], 'frames': {name: write(f'features/books/{name}.parquet', books[name])
                                                    for name in ('metrics', 'monthly')}}
    return section


def export_tuning(directories, write):
    from research.tuning_holdout import load_tuning_holdout
    studies = {}
    for value in directories:
        path = Path(value)
        study = load_tuning_holdout(path)
        check_tuning(study)
        trials = {key: write(f'tuning/{path.name}/{key}/trials.parquet',
                            pd.read_parquet(path / key / 'trials.parquet')) for key in study['metadata']['model_keys']}
        studies[path.name] = {'metadata': study['metadata'], 'source_metadata_sha256': digest_file(path / 'study.json'),
            'frames': {name: write(f'tuning/{path.name}/{name}.parquet', study[name])
                       for name in ('metrics', 'monthly', 'predictive')}, 'trials': trials}
    return {'studies': studies}


def export_final(directory, training_study, write, project_root):
    from research.final_books import load_final_books, window_books
    from research.basket_momentum import load_basket_study
    from research.basket_reporting import require_basket_training
    from research.evaluation import load_archived_test_results
    path = Path(directory)
    study = load_final_books(path)
    check_final(study)
    training = load_basket_study(training_study) if training_study else None
    if training is not None:
        require_basket_training(training)
    windows = window_books(study['monthly'], training['monthly'] if training else None)
    section = {'study_id': path.name, 'metadata': study['metadata'],
        'source_metadata_sha256': digest_file(path / 'study.json'),
        'frames': {name: write(f'final/{name}.parquet', study[name]) for name in ('metrics', 'monthly', 'fits')},
        'windows': {key: {name: write(f'final/windows/{key}/{name}.parquet', frame)
                          for name, frame in zip(('metrics', 'monthly'), frames)} for key, frames in windows.items()},
        'training_metadata': training['metadata'] if training else None}
    # Earlier non-ML comparisons are saved report CSVs, not source stock data.
    # Keep their viewing capability when those optional reports are available.
    try:
        archive = load_archived_test_results(project_root)
    except FileNotFoundError:
        archive = None
    if archive is not None:
        section['archive'] = {'frames': {}, 'values': {}}
        for name, value in archive.items():
            if isinstance(value, pd.DataFrame):
                section['archive']['frames'][name] = write(f'final/archive/{name}.parquet', value)
            else:
                section['archive']['values'][name] = value
    return section


class PublishedFeatures:
    def __init__(self, publication):
        self.publication = publication
        self.section = publication.metadata['tabs']['features']
        self.token = publication.token

    def list_studies(self):
        return [self.section['record']]

    def study(self):
        study = {'metadata': self.section['metadata'], 'method_notes': self.section['method_notes'],
            **{name: self.publication.frame(path) for name, path in self.section['frames'].items()},
            'summaries': {recipe: {**item['values'], **{name: self.publication.frame(path)
                for name, path in item['frames'].items()}} for recipe, item in self.section['summaries'].items()}}
        check_features(study)
        return study

    def book_studies(self):
        item = self.section.get('books')
        return [{'directory': Path(item['study_id']), 'metadata': item['metadata']}] if item else []

    def books(self):
        item = self.section['books']
        study = {'metadata': item['metadata'], **{name: self.publication.frame(path) for name, path in item['frames'].items()}}
        check_feature_books(study)
        return study


class PublishedTuning:
    def __init__(self, publication):
        self.publication = publication
        self.section = publication.metadata['tabs']['tuning']
        self.token = publication.token

    def list_studies(self):
        return [{'directory': Path(key), 'metadata': item['metadata']} for key, item in self.section['studies'].items()]

    def study(self, chosen):
        item = self.section['studies'][chosen['directory'].name]
        study = {'metadata': item['metadata'], **{name: self.publication.frame(path) for name, path in item['frames'].items()}}
        check_tuning(study)
        return study

    def trials(self, chosen, model):
        return self.publication.frame(self.section['studies'][chosen['directory'].name]['trials'][model])


class PublishedFinal:
    def __init__(self, publication):
        self.publication = publication
        self.section = publication.metadata['tabs']['final']
        self.token = publication.token

    def list_studies(self):
        return [{'directory': Path(self.section['study_id']), 'metadata': self.section['metadata']}]

    def study(self):
        study = {'metadata': self.section['metadata'], **{name: self.publication.frame(path) for name, path in self.section['frames'].items()}}
        check_final(study)
        return study

    def windows(self):
        return {key: tuple(self.publication.frame(item[name]) for name in ('metrics', 'monthly'))
                for key, item in self.section['windows'].items()}

    def training_metadata(self):
        return self.section['training_metadata']

    def archive(self):
        item = self.section.get('archive')
        return {**item['values'], **{name: self.publication.frame(path) for name, path in item['frames'].items()}} if item else None

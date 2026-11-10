"""Export completed presentation experiments; never fit or replay backtests."""
import argparse
from pathlib import Path

from published_results import export_strategies, export_grouping_comparison
from research.store import ExperimentStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extend-publication", type=Path, help="Existing publication to extend with --ward-study")
    parser.add_argument("--engineering-study", type=Path, help="Completed feature-engineering study, including final stage")
    parser.add_argument("--ward-study", type=Path, help="Saved Ward study with identical trading protocol")
    parser.add_argument("--run", help="Explicit saved training experiment ID")
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path, help="New immutable output directory")
    parser.add_argument("--momentum-study", help="Optional saved non-ML lookback comparison")
    parser.add_argument("--basic-run", nargs="+", default=[], help="Explicit saved Basic ML run IDs to publish")
    parser.add_argument("--basket-study", type=Path, help="Saved Advanced ML basket study directory")
    parser.add_argument("--feature-study", help="Saved registry feature-selection study ID")
    parser.add_argument("--feature-book-study", type=Path, help="Saved feature comparison for advanced basket portfolios")
    parser.add_argument("--tuning-study", type=Path, nargs="+", default=[], help="Saved training holdout directories")
    parser.add_argument("--final-study", type=Path, help="Saved frozen official-test study directory")
    args = parser.parse_args()
    if args.extend_publication:
        if not args.output or bool(args.ward_study) == bool(args.engineering_study):
            parser.error("--extend-publication requires one of --ward-study/--engineering-study, and a new --output")
        if args.engineering_study:
            from engineering_publication import export_engineering
            path = export_engineering(args.extend_publication, args.engineering_study, args.output)
        else:
            path = export_grouping_comparison(args.extend_publication, args.ward_study, args.output)
        print(f"Published saved grouping comparison: {path}")
        return
    if not args.run:
        parser.error("--run is required unless extending an existing publication")
    release = ("project" if args.feature_study or args.tuning_study or args.final_study else
               "advanced_ml" if args.basket_study else "basic_ml" if args.basic_run else "strategies")
    output = args.output or Path(__file__).resolve().parent / "published" / release
    path = export_strategies(ExperimentStore(args.artifact_root), args.run, output, args.momentum_study,
                             args.basic_run, args.basket_study, feature_study_id=args.feature_study,
                             feature_book_study=args.feature_book_study, tuning_studies=args.tuning_study,
                             final_book_study=args.final_study)
    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    print(f"Published results: {path}\n{size:,} bytes. No new strategy/model computation.")


if __name__ == "__main__":
    main()

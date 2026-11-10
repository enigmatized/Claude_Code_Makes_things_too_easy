"""Explicit training/tuning and frozen final evaluation; never called by the app."""
import argparse
from pathlib import Path
from research.legacy import PROJECT_ROOT
from research.engineering_study import create_engineering_study, run_engineering_stage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['training','final'],default='training')
    parser.add_argument('--study',type=Path,help='Resume the declared study; required for final')
    parser.add_argument('--recipes',nargs='+')
    parser.add_argument('--models',nargs='+')
    args=parser.parse_args()
    if args.phase=='final' and args.study is None:parser.error('Final requires --study from a completed training experiment.')
    if args.study:
        run_engineering_stage(PROJECT_ROOT,args.study,'test' if args.phase=='final' else 'training')
        print(args.study,flush=True)
    else:print(create_engineering_study(PROJECT_ROOT,args.recipes,args.models),flush=True)


if __name__=='__main__':main()

"""Run additional geographical and robustness analyses after run_all.py."""
import argparse
from src.extended_analysis import run

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Regional analysis and robustness checks')
    parser.add_argument('--large-sample',action='store_true',help='Also run 400k and full complete-case samples (high RAM use)')
    parser.add_argument('--rebuild-56',action='store_true',help='Reconstruct 56-day panel from historical Parquet (slower, high RAM use)')
    args=parser.parse_args()
    run(large=args.large_sample,rebuild56=args.rebuild_56)

"""Extract the BinResults patch-pair dataset (resumable network job).

Usage:
  python extract_binpairs.py [--root //192.168.4.81/VQ-Data/BinResults]
        [--out datasets/binpairs] [--folders 1500] [--patches 10] [--seed 0]
"""

import argparse

from adaptive_filters.binpairs import extract


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="//192.168.4.81/VQ-Data/BinResults")
    ap.add_argument("--out", default=r"datasets\binpairs")
    ap.add_argument("--folders", type=int, default=1500)
    ap.add_argument("--patches", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    extract(args.root, args.out, n_folders=args.folders,
            patches_per_pair=args.patches, seed=args.seed)


if __name__ == "__main__":
    main()

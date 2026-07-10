"""Generate the on-disk training-pair dataset (degraded, pristine) from
real codecs: H.264 / HEVC / MPEG-2 with randomized rate points, profiles
and tools, plus a packet-loss family with codec-aware corruption.

Usage:
  python generate_pairs.py [--out datasets/pairs]
        [--sources F:\\DVI\\BVI-DVC\\Videos F:\\DVI\\BVI-CC1\\ORIG_MP4]
        [--clips 60] [--patches-per-frame 20] [--seed 0]

Per clip: one compression shard + one loss shard, codec cycled
h264 -> hevc -> mpeg2. Patches are 256x256 uint8 YUV (sub-crop for the
smaller filter generations at load time).
"""

import argparse

from adaptive_filters.pairgen import generate_pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=r"datasets\pairs")
    ap.add_argument("--sources", nargs="+",
                    default=[r"F:\DVI\BVI-DVC\Videos",
                             r"F:\DVI\BVI-CC1\ORIG_MP4"])
    ap.add_argument("--clips", type=int, default=60)
    ap.add_argument("--patches-per-frame", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    generate_pairs(args.sources, args.out, clips=args.clips,
                   patches_per_frame=args.patches_per_frame, seed=args.seed)


if __name__ == "__main__":
    main()

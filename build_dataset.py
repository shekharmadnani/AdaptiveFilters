"""Build a feature/label dataset from a manifest of encodes.

Usage:
  python build_dataset.py manifest.json dataset.npz [--step 6] [--no-ref]
                          [--ldct models/kdct.pt] [--cache CACHE_DIR]

See adaptive_filters/dataset.py for the manifest format. VMAF is computed
via ffmpeg's libvmaf when no pre-computed vmaf_json is given (and persisted
to the manifest's vmaf_json path so interrupted runs resume). --cache makes
per-video feature extraction resumable too.
"""

import argparse

from adaptive_filters.dataset import build_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", help="manifest JSON (list of entries)")
    ap.add_argument("out", help="output .npz path")
    ap.add_argument("--step", type=int, default=6,
                    help="sample every Nth frame (default 6)")
    ap.add_argument("--no-ref", action="store_true",
                    help="skip featurizing reference frames (anchor corpus)")
    ap.add_argument("--ldct", default=None,
                    help="weights path: add the learned DCT probe (PyTorch)")
    ap.add_argument("--lkm", default=None,
                    help="weights path: add the learned K-map probe (PyTorch)")
    ap.add_argument("--cache", default=None,
                    help="directory for per-video feature caches (resume). "
                         "Use a DISTINCT dir per probe configuration -- the "
                         "cache is keyed by video, not by feature set.")
    args = ap.parse_args()

    extractor = None
    if args.ldct or args.lkm:
        from adaptive_filters.pipeline import FeatureExtractor, default_probes

        probes = default_probes()
        if args.ldct:
            from adaptive_filters.probes.learned_dct_probe import LearnedDctProbe
            probes.append(LearnedDctProbe(args.ldct))
        if args.lkm:
            from adaptive_filters.probes.learned_kmap_probe import LearnedKMapProbe
            probes.append(LearnedKMapProbe(args.lkm))
        extractor = FeatureExtractor(probes=probes)

    build_dataset(args.manifest, args.out, frame_step=args.step,
                  include_ref=not args.no_ref, extractor=extractor,
                  cache_dir=args.cache)


if __name__ == "__main__":
    main()

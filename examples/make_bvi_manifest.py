"""Generate a manifest for the BVI-CC1 HD codec-comparison set.

Pairs each encode in Test_MP4/CC-HD with its master in ORIG_MP4 by scene
name; content id = scene token (drives the grouped split). VMAF logs are
assigned per-encode paths so an interrupted dataset build resumes.

Usage:
  python examples/make_bvi_manifest.py [--bvi F:\\DVI\\BVI-CC1]
        [--out datasets/bvi_cc_hd/manifest.json] [--limit N]
"""

import argparse
import json
import os
import re

NAME_RE = re.compile(r"^(S\d+[A-Za-z0-9]+)_(1920x1080[^_]*(?:_[^_]+)*?)_"
                     r"(AV1|HM|VTM\w*)(_.*)?\.mp4$")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bvi", default=r"F:\DVI\BVI-CC1")
    ap.add_argument("--out",
                    default=os.path.join("datasets", "bvi_cc_hd",
                                         "manifest.json"))
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N encodes (smoke tests)")
    args = ap.parse_args()

    orig_dir = os.path.join(args.bvi, "ORIG_MP4")
    test_dir = os.path.join(args.bvi, "Test_MP4", "CC-HD")
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    vmaf_dir = os.path.join(out_dir, "vmaf_logs")

    masters = {}
    for f in os.listdir(orig_dir):
        if f.endswith(".mp4") and "1920x1080" in f:
            scene = f.split("_")[0]  # e.g. S11AirAcrobatic
            masters[scene] = os.path.join(orig_dir, f)

    entries, skipped = [], 0
    for f in sorted(os.listdir(test_dir)):
        if not f.endswith(".mp4"):
            continue
        scene = f.split("_")[0]
        ref = masters.get(scene)
        if ref is None:
            skipped += 1
            continue
        entries.append({
            "content": scene,
            "dist": os.path.join(test_dir, f),
            "ref": ref,
            "vmaf_json": os.path.join(vmaf_dir, f + ".vmaf.json"),
        })
    if args.limit:
        entries = entries[: args.limit]

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=1)
    contents = sorted({e["content"] for e in entries})
    print(f"manifest -> {args.out}")
    print(f"  encodes: {len(entries)}  contents: {len(contents)}  "
          f"skipped (no master): {skipped}")
    print(f"  contents: {', '.join(contents)}")


if __name__ == "__main__":
    main()

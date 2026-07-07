"""Generate a miniature real encode ladder for end-to-end pipeline testing.

Uses ffmpeg (libx264) to create 3 synthetic source contents, encode each at
4 CRF levels, and write a manifest.json for build_dataset.py. VMAF labels
are computed later by the dataset builder via libvmaf.

Usage:  python examples/build_demo_dataset.py [--out examples/demo_data]
"""

import argparse
import json
import os
import subprocess

SIZE = "320x240"
RATE = 24
DURATION = 3
CRFS = [18, 28, 38, 46]

CONTENTS = {
    "c_testsrc2": f"testsrc2=size={SIZE}:rate={RATE}",
    "c_mandelbrot": f"mandelbrot=size={SIZE}:rate={RATE}",
    "c_grain": f"testsrc2=size={SIZE}:rate={RATE},noise=alls=12:allf=t+u",
}


def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join("examples", "demo_data"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    manifest = []
    for name, src in CONTENTS.items():
        ref = os.path.join(args.out, f"{name}.y4m")
        if not os.path.exists(ref):
            print(f"generating {ref}")
            run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", src,
                 "-t", str(DURATION), "-pix_fmt", "yuv420p", ref])
        for crf in CRFS:
            dist = os.path.join(args.out, f"{name}_crf{crf}.mp4")
            if not os.path.exists(dist):
                print(f"encoding  {dist}")
                run(["ffmpeg", "-y", "-v", "error", "-i", ref,
                     "-c:v", "libx264", "-preset", "veryfast",
                     "-crf", str(crf), "-pix_fmt", "yuv420p", dist])
            manifest.append({
                "content": name,
                "dist": os.path.basename(dist),
                "ref": os.path.basename(ref),
            })

    mpath = os.path.join(args.out, "manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"manifest -> {mpath}  ({len(manifest)} encodes, "
          f"{len(CONTENTS)} contents)")


if __name__ == "__main__":
    main()

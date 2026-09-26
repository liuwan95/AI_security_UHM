#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from randomized_smoothing.architectures import ResNetCifar
from randomized_smoothing.core import ABSTAIN, max_certifiable_radius
from randomized_smoothing.data import load_cifar10
from randomized_smoothing.smooth import SmoothClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sigma", type=float, default=None,
                   help="smoothing sigma; defaults to the checkpoint's")
    p.add_argument("--n0", type=int, default=100, help="selection samples")
    p.add_argument("--n", type=int, default=100_000, help="estimation samples")
    p.add_argument("--alpha", type=float, default=0.001)
    p.add_argument("--skip", type=int, default=20, help="certify every k-th image")
    p.add_argument("--max", type=int, default=500, help="max images to certify")
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--data-dir", type=Path, default=Path("./data"))
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sigma = args.sigma if args.sigma is not None else ckpt["sigma"]
    model = ResNetCifar(ckpt["depth"]).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(ckpt["state_dict"])

    gen = torch.Generator(device=device).manual_seed(args.seed)
    smoothed = SmoothClassifier(model, 10, sigma, device=device,
                                batch_size=args.batch_size, generator=gen)
    data = load_cifar10(args.data_dir, device)

    ceiling = max_certifiable_radius(sigma, args.n, args.alpha)
    print(f"sigma={sigma} n0={args.n0} n={args.n} alpha={args.alpha}\n"
          f"radius ceiling at this sampling budget: {ceiling:.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    indices = list(range(0, data.x_test.shape[0], args.skip))[: args.max]
    t_all = time.time()
    with args.out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["idx", "label", "predict", "radius", "pA_lower",
                         "correct", "abstain", "seconds"])
        for k, i in enumerate(indices):
            t0 = time.time()
            r = smoothed.certify(data.x_test[i], args.n0, args.n, args.alpha)
            dt = time.time() - t0
            label = int(data.y_test[i].item())
            correct = int(r.prediction == label and not r.abstained)
            writer.writerow([i, label, r.prediction, f"{r.radius:.6f}",
                             f"{r.lower_probability:.6f}", correct,
                             int(r.abstained), f"{dt:.3f}"])
            if (k + 1) % 25 == 0 or k == 0:
                print(f"[{k+1:>4}/{len(indices)}] idx={i:>5} label={label} "
                      f"pred={r.prediction} R={r.radius:.3f} ({dt:.2f}s)", flush=True)
    print(f"wrote {args.out}  ({len(indices)} images, {time.time()-t_all:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()

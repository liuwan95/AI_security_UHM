#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from randomized_smoothing.architectures import ResNetCifar
from randomized_smoothing.data import augment, load_cifar10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sigma", type=float, required=True,
                   help="noise level this model will be smoothed with")
    p.add_argument("--train-sigma", type=float, default=None,
                   help="training noise; defaults to --sigma. Set 0 for the "
                        "clean-training ablation.")
    p.add_argument("--depth", type=int, default=110)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--lr-step", type=int, default=50)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--data-dir", type=Path, default=Path("./data"))
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def evaluate(model, x, y, sigma, generator, batch_size=1000) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = x[i : i + batch_size]
            if sigma > 0:
                xb = xb + torch.randn(xb.shape, device=xb.device,
                                      generator=generator) * sigma
            with torch.autocast("cuda", torch.float16,
                                enabled=xb.device.type == "cuda"):
                correct += (model(xb).argmax(1) == y[i : i + batch_size]).sum().item()
    model.train()
    return correct / x.shape[0]


def main() -> None:
    args = parse_args()
    train_sigma = args.sigma if args.train_sigma is None else args.train_sigma
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    gen = torch.Generator(device=device).manual_seed(args.seed)
    eval_gen = torch.Generator(device=device).manual_seed(args.seed + 1)

    data = load_cifar10(args.data_dir, device)
    model = ResNetCifar(args.depth).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, args.lr_step, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda")

    n = len(data)
    steps = n // args.batch_size
    print(f"train sigma={train_sigma} smooth sigma={args.sigma} depth={args.depth} "
          f"epochs={args.epochs} device={device}", flush=True)

    history, t_start = [], time.time()
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(n, device=device, generator=gen)
        running = torch.zeros((), device=device)
        for s in range(steps):
            idx = order[s * args.batch_size : (s + 1) * args.batch_size]
            xb = augment(data.x_train[idx], gen)
            if train_sigma > 0:
                xb = xb + torch.randn(xb.shape, device=device, generator=gen) * train_sigma
            yb = data.y_train[idx]
            with torch.autocast("cuda", torch.float16):
                loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.detach()
        sched.step()
        mean_loss = (running / steps).item()

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            clean = evaluate(model, data.x_test, data.y_test, 0.0, eval_gen)
            noisy = evaluate(model, data.x_test, data.y_test, args.sigma, eval_gen)
            history.append({"epoch": epoch + 1, "loss": mean_loss,
                            "test_clean": clean, "test_noisy": noisy})
            print(f"epoch {epoch+1:3d}/{args.epochs}  loss {mean_loss:.4f}  "
                  f"clean {clean:.4f}  noisy(sigma={args.sigma}) {noisy:.4f}  "
                  f"[{time.time()-t_start:.0f}s]", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "depth": args.depth,
                "sigma": args.sigma, "train_sigma": train_sigma,
                "args": vars(args) | {"data_dir": str(args.data_dir),
                                      "out": str(args.out)},
                "history": history, "train_seconds": time.time() - t_start},
               args.out)
    args.out.with_suffix(".json").write_text(json.dumps(
        {"sigma": args.sigma, "train_sigma": train_sigma, "depth": args.depth,
         "epochs": args.epochs, "history": history,
         "train_seconds": time.time() - t_start}, indent=2))
    print(f"saved {args.out}  ({time.time()-t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

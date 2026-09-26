from __future__ import annotations

import numpy as np
import torch

from .core import (
    CertificationResult,
    PredictionResult,
    certify_from_counts,
    predict_from_counts,
)


class SmoothClassifier:
    def __init__(
        self,
        base_classifier: torch.nn.Module,
        num_classes: int,
        sigma: float,
        *,
        device: str = "cuda",
        batch_size: int = 1000,
        generator: torch.Generator | None = None,
    ) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.base_classifier = base_classifier.to(device).eval()
        self.num_classes = num_classes
        self.sigma = float(sigma)
        self.device = device
        self.batch_size = batch_size
        self.generator = generator

    @torch.no_grad()
    def sample_counts(self, x: torch.Tensor, num_samples: int) -> np.ndarray:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        x = x.to(self.device)
        counts = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)

        remaining = num_samples
        while remaining:
            current = min(self.batch_size, remaining)
            batch = x.unsqueeze(0).expand(current, *x.shape)
            noise = torch.randn(
                batch.shape,
                device=self.device,
                generator=self.generator,
            ) * self.sigma
            noisy = (batch + noise).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", torch.float16, enabled=self.device == "cuda"):
                logits = self.base_classifier(noisy)
            counts += torch.bincount(
                logits.argmax(1), minlength=self.num_classes
            )
            remaining -= current
        return counts.cpu().numpy()

    def predict(self, x: torch.Tensor, n: int, alpha: float) -> PredictionResult:
        return predict_from_counts(self.sample_counts(x, n), alpha)

    def certify(
        self, x: torch.Tensor, n0: int, n: int, alpha: float
    ) -> CertificationResult:
        selection = self.sample_counts(x, n0)
        estimation = self.sample_counts(x, n)
        return certify_from_counts(selection, estimation, self.sigma, alpha)

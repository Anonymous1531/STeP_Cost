#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List


class TwoComponentGMM:
    """One-dimensional two-component GMM used for category-relative motion labels."""

    def __init__(self, min_samples: int = 10) -> None:
        self.min_samples = int(min_samples)
        self.samples: List[float] = []
        self.mu = [0.0, 0.0]
        self.sigma = [1.0, 1.0]
        self.pi = [0.5, 0.5]
        self.fitted = False

    @property
    def is_ready(self) -> bool:
        return self.fitted

    def add_sample(self, speed_mps: float) -> None:
        speed = float(speed_mps)
        if not math.isfinite(speed) or speed < 0.0:
            return
        self.samples.append(speed)
        if len(self.samples) >= self.min_samples:
            self.fit()

    @staticmethod
    def _gaussian(x: float, mu: float, sigma: float) -> float:
        sigma = max(1e-6, float(sigma))
        return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (
            sigma * math.sqrt(2.0 * math.pi)
        )

    def fit(self) -> None:
        x = list(self.samples)
        n = len(x)
        if n < 2:
            return

        sorted_x = sorted(x)
        median = sorted_x[n // 2]
        slow_vals = [v for v in x if v <= median]
        fast_vals = [v for v in x if v > median]

        mu0 = sum(slow_vals) / max(1, len(slow_vals))
        mu1 = sum(fast_vals) / max(1, len(fast_vals))
        if mu0 >= mu1:
            mu0 = sorted_x[n // 4]
            mu1 = sorted_x[(3 * n) // 4]

        sigma0 = sigma1 = max(0.01, (mu1 - mu0) / 4.0)
        pi0 = pi1 = 0.5

        for _ in range(50):
            r0: List[float] = []
            r1: List[float] = []
            for v in x:
                p0 = pi0 * self._gaussian(v, mu0, sigma0)
                p1 = pi1 * self._gaussian(v, mu1, sigma1)
                total = p0 + p1 + 1e-12
                r0.append(p0 / total)
                r1.append(p1 / total)

            n0 = sum(r0) + 1e-12
            n1 = sum(r1) + 1e-12
            mu0_new = sum(r * v for r, v in zip(r0, x)) / n0
            mu1_new = sum(r * v for r, v in zip(r1, x)) / n1
            sigma0_new = math.sqrt(
                sum(r * (v - mu0_new) ** 2 for r, v in zip(r0, x)) / n0
            ) + 1e-4
            sigma1_new = math.sqrt(
                sum(r * (v - mu1_new) ** 2 for r, v in zip(r1, x)) / n1
            ) + 1e-4

            converged = (
                abs(mu0_new - mu0) < 1e-5
                and abs(mu1_new - mu1) < 1e-5
            )
            mu0, mu1 = mu0_new, mu1_new
            sigma0, sigma1 = sigma0_new, sigma1_new
            pi0, pi1 = n0 / n, n1 / n
            if converged:
                break

        if mu0 > mu1:
            mu0, mu1 = mu1, mu0
            sigma0, sigma1 = sigma1, sigma0
            pi0, pi1 = pi1, pi0

        self.mu = [mu0, mu1]
        self.sigma = [sigma0, sigma1]
        self.pi = [pi0, pi1]
        self.fitted = True

    def predict(self, speed_mps: float) -> str:
        if not self.fitted:
            return "fast"
        v = float(speed_mps)
        q_slow = self.pi[0] * self._gaussian(v, self.mu[0], self.sigma[0])
        q_fast = self.pi[1] * self._gaussian(v, self.mu[1], self.sigma[1])
        return "slow" if q_slow > q_fast else "fast"

    def to_dict(self) -> dict:
        return {
            "samples": self.samples,
            "fitted": self.fitted,
            "mu": self.mu,
            "sigma": self.sigma,
            "pi": self.pi,
        }

    @classmethod
    def from_dict(cls, data: dict, min_samples: int = 10) -> "TwoComponentGMM":
        obj = cls(min_samples=min_samples)
        obj.samples = [float(v) for v in data.get("samples", [])]
        obj.fitted = bool(data.get("fitted", False))
        obj.mu = [float(v) for v in data.get("mu", [0.0, 0.0])]
        obj.sigma = [float(v) for v in data.get("sigma", [1.0, 1.0])]
        obj.pi = [float(v) for v in data.get("pi", [0.5, 0.5])]
        return obj


class CategoryConditionedGMM:
    """Separate two-component GMM for each base semantic category."""

    def __init__(self, min_samples: int = 10) -> None:
        self.min_samples = int(min_samples)
        self.models: Dict[str, TwoComponentGMM] = {}

    def _model(self, base_tag: str) -> TwoComponentGMM:
        key = str(base_tag).strip().lower()
        if key not in self.models:
            self.models[key] = TwoComponentGMM(self.min_samples)
        return self.models[key]

    def add_sample(self, base_tag: str, speed_mps: float) -> None:
        self._model(base_tag).add_sample(speed_mps)

    def is_ready(self, base_tag: str) -> bool:
        return self._model(base_tag).is_ready

    def predict(self, base_tag: str, speed_mps: float) -> str:
        return self._model(base_tag).predict(speed_mps)

    def load(self, path: str) -> None:
        p = Path(path).expanduser()
        if not p.exists():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return
        for tag, row in data.items():
            if isinstance(row, dict):
                self.models[tag] = TwoComponentGMM.from_dict(row, self.min_samples)

    def save(self, path: str) -> None:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({tag: model.to_dict() for tag, model in self.models.items()}, indent=2),
            encoding="utf-8",
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Inspect or query the category-conditioned GMM motion classifier used by STeP-Cost."
    )
    ap.add_argument("--samples", required=True, help="Path to gmm_samples.json")
    ap.add_argument("--tag", required=True, help="Base semantic tag, e.g. person")
    ap.add_argument("--speed", type=float, required=True, help="Tracked obstacle speed in m/s")
    ap.add_argument("--min-samples", type=int, default=10)
    args = ap.parse_args()

    model = CategoryConditionedGMM(min_samples=args.min_samples)
    model.load(args.samples)
    ready = model.is_ready(args.tag)
    result = {
        "tag_key": args.tag,
        "speed_mps": args.speed,
        "gmm_ready": ready,
        "speed_class": model.predict(args.tag, args.speed) if ready else None,
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

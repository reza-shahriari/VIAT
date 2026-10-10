"""Random-frame audit statistics for the Autopilot pipeline.

The review queue finds errors; only a random sample can say how many remain in
what nobody looked at. With exact one-sided Clopper-Pearson bounds, reviewing n
random frames and finding k with an error certifies (at the given confidence) that
at most ``upper_bound(k, n)`` of all frames contain an error. For example 0 errors in
59 frames certifies <= 5 %, 0 in 299 certifies <= 1 % (95 % confidence).
"""
import math
import random


def _binom_cdf(k, n, p):
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    total = 0.0
    for i in range(0, k + 1):
        total += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
    return min(1.0, total)


def upper_bound(k, n, confidence=0.95):
    """One-sided Clopper-Pearson upper bound on an error rate after k errors in n samples."""
    if n <= 0 or k >= n:
        return 1.0
    alpha = 1.0 - confidence
    lo, hi = k / float(n), 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _binom_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def samples_needed(target, k=0, confidence=0.95, limit=100000):
    """Smallest n at which k observed errors would still certify ``<= target``."""
    n = max(k + 1, 1)
    # coarse doubling, then binary search (upper_bound decreases with n)
    hi = n
    while upper_bound(k, hi, confidence) > target:
        hi *= 2
        if hi > limit:
            return None
    lo = max(n, hi // 2)
    while lo < hi:
        mid = (lo + hi) // 2
        if upper_bound(k, mid, confidence) <= target:
            hi = mid
        else:
            lo = mid + 1
    return hi


def sample_frames(frames, n, seed=None, exclude=()):
    pool = [f for f in frames if f not in set(exclude)]
    rng = random.Random(seed)
    return sorted(rng.sample(pool, min(n, len(pool))))

"""Face-verification metrics with identity-level (cluster) bootstrap."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def threshold_at_far(imp, far, w=None):
    """Score t such that a (weighted) fraction `far` of impostors score above t."""
    order = np.argsort(imp)
    s = imp[order]
    w = np.ones_like(s) if w is None else w[order]
    cdf = np.cumsum(w) / w.sum()
    return s[min(np.searchsorted(cdf, 1 - far), len(s) - 1)]


def tar_at_far(scores, labels, far, w=None):
    g, i = labels == 1, labels == 0
    wg = None if w is None else w[g]
    wi = None if w is None else w[i]
    t = threshold_at_far(scores[i], far, wi)
    acc = scores[g] > t
    return float(acc.mean() if wg is None else (acc * wg).sum() / wg.sum()), float(t)


def tar_fn(scores, labels, far):
    """Fast weighted TAR@FAR for bootstrapping: impostors are sorted once. Returns f(w) -> TAR."""
    g = labels == 1
    gen, imp = scores[g], scores[~g]
    order = np.argsort(imp)
    imp_s = imp[order]

    def f(w=None):
        wi = np.ones(len(imp_s)) if w is None else w[~g][order]
        cdf = np.cumsum(wi) / wi.sum()
        t = imp_s[min(np.searchsorted(cdf, 1 - far), len(imp_s) - 1)]
        acc = gen > t
        return float(acc.mean() if w is None else (acc * w[g]).sum() / w[g].sum())
    return f


def eer(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    j = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[j] + fnr[j]) / 2)


def auc(scores, labels):
    return float(roc_auc_score(labels, scores))


def rate_above(scores, t, w=None):
    a = scores > t
    return float(a.mean() if w is None else (a * w).sum() / w.sum())


def _weights(owner, rng):
    u, inv = np.unique(owner, return_inverse=True)
    counts = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
    return counts[inv].astype(np.float64)


def bootstrap(fn, owner, n=200, seed=0):
    """95% CI of fn(weights) where clusters (cats) are resampled with replacement."""
    rng = np.random.default_rng(seed)
    vals = [fn(_weights(owner, rng)) for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_diff(fn_a, fn_b, owner, n=200, seed=0):
    """Paired cluster bootstrap of fn_a - fn_b; returns (diff, lo, hi). Significant if CI excludes 0."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        w = _weights(owner, rng)
        d.append(fn_a(w) - fn_b(w))
    return float(fn_a(None) - fn_b(None)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def det_points(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr, 1 - tpr

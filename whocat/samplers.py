"""P x K identity batch samplers: random vs coat-aware."""
import numpy as np


class _Queue:
    def __init__(self, items, rng):
        self.items, self.rng, self.q = np.asarray(items), rng, []

    def take(self, n):
        out = []
        while len(out) < n:
            if not len(self.q):
                self.q = list(self.rng.permutation(self.items))
            out.append(self.q.pop())
        return out


class PKSampler:
    """Yields batches of P identities x K images.

    coat_groups=None : identities drawn uniformly at random (standard batches)
    coat_groups=dict : each batch holds G coat groups with P/G cats each, so most in-batch
                       negatives share the anchor's coat ('unknown' coat is its own group)
    """

    def __init__(self, imgs_by_label, P=64, K=2, coat_groups=None, G=4, seed=0):
        self.rng = np.random.default_rng(seed)
        self.imgs, self.P, self.K, self.G = imgs_by_label, P, K, G
        self.n_batches = len(imgs_by_label) // P
        if coat_groups is None:
            self.queues = None
            self.all = _Queue(np.arange(len(imgs_by_label)), self.rng)
        else:
            groups = {c: v for c, v in coat_groups.items() if len(v) >= P // G}
            self.names = list(groups)
            self.queues = {c: _Queue(v, self.rng) for c, v in groups.items()}
            sizes = np.array([len(groups[c]) for c in self.names], float)
            self.p = sizes / sizes.sum()

    def __len__(self):
        return self.n_batches

    def _labels(self):
        if self.queues is None:
            return self.all.take(self.P)
        g = self.rng.choice(self.names, size=min(self.G, len(self.names)), replace=False, p=self.p)
        per = self.P // len(g)
        out = []
        for c in g:
            out += self.queues[c].take(per)
        return out

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for lab in self._labels():
                imgs = self.imgs[lab]
                batch += list(self.rng.choice(imgs, self.K, replace=len(imgs) < self.K))
            yield batch

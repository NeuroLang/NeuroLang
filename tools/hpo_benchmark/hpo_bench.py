"""Phase 1 of the HPO benchmark: models, noise, scoring (spec 2026-11-23-hpo-benchmark-design)."""
import collections, random
import numpy as np
from hpo_lib import load as load_facts
from hpo_onto import load as load_onto, closure
from hpo_cases import load_cases

GENERIC = {"HP:0000001", "HP:0000118"}
SPUR_LEVELS, DROP_LEVELS = (0, 1, 2, 4), (0.0, 0.25, 0.5)


class World:
    def __init__(self):
        self.facts, _ = load_facts()
        parents, self.obsolete, self.replaced = load_onto()
        self.A = closure(parents)
        self.diseases = sorted({d for d, _ in self.facts})
        self.di = {d: i for i, d in enumerate(self.diseases)}
        self.n = len(self.diseases)
        self.by_pheno = collections.defaultdict(list)
        self.by_dis = collections.defaultdict(dict)
        for (d, s), p in self.facts.items():
            self.by_pheno[s].append((self.di[d], p))
            self.by_dis[d][s] = p
        self.phenos = sorted(self.by_pheno)
        self.desc = collections.defaultdict(dict)
        for t, anc in self.A.items():
            for a, dist in anc.items():
                self.desc[a][t] = dist
        self._cache = {}

    def vec(self, t, k_up, k_down, gamma):
        """p(d, t) for every disease: exact, plus gamma-discounted ancestor/descendant matches."""
        key = (t, k_up, k_down, gamma)
        v = self._cache.get(key)
        if v is None:
            v = np.zeros(self.n)
            for i, p in self.by_pheno.get(t, ()):
                v[i] = p
            if k_up:
                for a, dist in self.A.get(t, {}).items():
                    if dist <= k_up and a not in GENERIC:
                        for i, p in self.by_pheno.get(a, ()):
                            v[i] = max(v[i], gamma * p)
            if k_down and t not in GENERIC:
                for s, dist in self.desc.get(t, {}).items():
                    if dist <= k_down:
                        for i, p in self.by_pheno.get(s, ()):
                            v[i] = max(v[i], gamma * p)
            if len(self._cache) > 200000:
                self._cache.clear()
            self._cache[key] = v
        return v

    def probs(self, obs, cfg):
        """Matrix (len(obs), n) of the per-term probabilities of cfg."""
        M = np.stack([self.vec(t, cfg["k_up"], cfg["k_down"], cfg["gamma"]) for t in obs])
        if cfg["eps"]:
            M = 1 - (1 - M) * (1 - cfg["eps"])
        return M


def posterior(W, obs, cfg):
    """Uniform-prior posterior over diseases (numpy reference); zeros if empty."""
    M = W.probs(obs, cfg)
    with np.errstate(divide="ignore"):
        ll = np.log(M).sum(axis=0)
    if np.all(np.isinf(ll)):
        return np.zeros(W.n)
    w = np.exp(ll - ll.max())
    return w / w.sum()


def overlap_scores(W, obs):
    """Overlap baseline: sum of annotation frequencies, tie-broken by matched terms."""
    M = W.probs(obs, dict(k_up=0, k_down=0, gamma=1.0, eps=0.0))
    return M.sum(axis=0) + 1e-6 * (M > 0).sum(axis=0)


def rank_of(scores, i, tol=1e-12):
    """Mean rank of disease i among ties; None if its score is 0 (not supported)."""
    if scores[i] <= 0:
        return None
    s = scores[i]
    gt = int((scores > s * (1 + tol)).sum())
    eq = int((np.abs(scores - s) <= tol * s).sum())
    return gt + (eq + 1) / 2


def perturb(W, obs, spur, drop, rng):
    obs = list(obs)
    n_drop = min(int(round(drop * len(obs))), max(len(obs) - 2, 0))
    rng.shuffle(obs)
    obs = obs[n_drop:]
    obs += rng.sample(W.phenos, spur)
    return sorted(set(obs))


def simulate(W, rng, k):
    pool = [d for d in W.diseases if len(W.by_dis[d]) >= 6]
    while True:
        d = rng.choice(pool)
        present = [s for s, p in W.by_dis[d].items() if rng.random() < p]
        if len(present) >= k:
            return d, sorted(rng.sample(present, k))


def split_diseases(cases, dev_frac=0.3, seed=0):
    dis = sorted({d for d, _ in cases})
    rng = random.Random(seed)
    rng.shuffle(dis)
    n_dev = int(round(dev_frac * len(dis)))
    return set(dis[:n_dev]), set(dis[n_dev:])

"""One-time script: generates the faiss e-means reference fixture used by
test/emeans.faiss-fixture.test.ts. Not run in CI and not a build dependency --
only re-run this if the fixture's input data or options change.

Prerequisites:
    pip install faiss-cpu numpy

Run:
    python scripts/generate-faiss-fixture.py
"""
import json

import numpy as np
from faiss.contrib.e_means import NumpyEmeans

rng = np.random.RandomState(42)
centers = np.array([[0, 0], [10, 10], [0, 10], [10, 0]], dtype="float32")
data = np.concatenate(
    [c + rng.normal(scale=0.25, size=(200, 2)).astype("float32") for c in centers]
)

model = NumpyEmeans(d=2, k=4, nepochs=5, seed=42)
centroids, labels = model.train(data)

fixture = {
    "data": data.tolist(),
    "dim": 2,
    "k": 4,
    "nepochs": 5,
    "seed": 42,
    "finalLoss": float(model.obj[-1]),
    "lossPerEpoch": model.obj.tolist(),
}

with open("test/fixtures/emeans-faiss-reference.json", "w") as f:
    json.dump(fixture, f, indent=2)

print("Wrote test/fixtures/emeans-faiss-reference.json")
print("Loss per epoch:", fixture["lossPerEpoch"])

# --- Harder fixture ---------------------------------------------------------
# The fixture above is too easy to discriminate correct vs. wrong implementations:
# inter-cluster separation (10 units) is ~40x the per-cluster noise std (0.25), so
# almost any non-catastrophic clustering lands at nearly the same loss. It's also
# d=2, which means NO_REVIVAL_MAX_D disables dead-cluster revival on both the faiss
# side and our port's side, so reviveUnused()/split logic is never exercised.
#
# This harder fixture uses d=8 (> NO_REVIVAL_MAX_D, so revival is active on both
# sides), 6 true Gaussian blobs with centers only ~1.4-3.0 noise stds apart
# (meaningfully overlapping, not ~40 sigma apart), and k=8 (more clusters than true
# structure) so dead-cluster revival actually fires during training, not just in
# principle. See task-7-report.md for the empirical validation of the tolerance
# chosen in the accompanying test.
hard_dim = 8
hard_k_true = 6
hard_noise_std = 1.0
hard_center_scale = 0.625
hard_n_per_cluster = 200
hard_k = 8

hard_rng = np.random.RandomState(42)
hard_centers = hard_rng.normal(
    scale=hard_center_scale, size=(hard_k_true, hard_dim)
).astype("float32")
hard_data = np.concatenate(
    [
        c + hard_rng.normal(scale=hard_noise_std, size=(hard_n_per_cluster, hard_dim)).astype("float32")
        for c in hard_centers
    ]
)

hard_model = NumpyEmeans(d=hard_dim, k=hard_k, nepochs=5, seed=42)
hard_centroids, hard_labels = hard_model.train(hard_data)

hard_fixture = {
    "data": hard_data.tolist(),
    "dim": hard_dim,
    "k": hard_k,
    "nepochs": 5,
    "seed": 42,
    "finalLoss": float(hard_model.obj[-1]),
    "lossPerEpoch": hard_model.obj.tolist(),
}

with open("test/fixtures/emeans-faiss-reference-hard.json", "w") as f:
    json.dump(hard_fixture, f, indent=2)

print("Wrote test/fixtures/emeans-faiss-reference-hard.json")
print("Hard loss per epoch:", hard_fixture["lossPerEpoch"])
print("Hard cluster usage counts:", np.bincount(hard_labels, minlength=hard_k))

# --- High-dim fixture (exercises the 'throughput' batch-law branch) --------
# resolveBatchLaw('auto', dim) in src/emeansSchedule.ts picks 'quality' for
# dim <= AUTO_QUALITY_MAX_D (16) and 'throughput' above that, independent of
# n. Both fixtures above are dim=2 and dim=8, so neither ever exercises the
# 'throughput' branch -- yet every realistic embedding dimension the README
# advertises (50-3072) runs exclusively through 'throughput'. This fixture
# uses dim=32 (> 16, so 'throughput' is selected unconditionally) with the
# same "meaningfully overlapping" cluster design as the hard fixture above
# (not ~40-sigma separated) and k=8 (> true cluster count) so dead-cluster
# revival is active (dim=32 > NO_REVIVAL_MAX_D) and n is large enough for
# multiple mini-batches per epoch at the batch sizes resolveRecipe computes
# for this (n, k, nepochs) under the throughput law.
highdim_dim = 32
highdim_k_true = 6
highdim_noise_std = 0.8
highdim_center_scale = 0.3
highdim_n_per_cluster = 200
highdim_k = 8

highdim_rng = np.random.RandomState(42)
highdim_centers = highdim_rng.normal(
    scale=highdim_center_scale, size=(highdim_k_true, highdim_dim)
).astype("float32")
highdim_data = np.concatenate(
    [
        c + highdim_rng.normal(scale=highdim_noise_std, size=(highdim_n_per_cluster, highdim_dim)).astype("float32")
        for c in highdim_centers
    ]
)

highdim_model = NumpyEmeans(d=highdim_dim, k=highdim_k, nepochs=5, seed=42)
highdim_centroids, highdim_labels = highdim_model.train(highdim_data)

highdim_fixture = {
    "data": highdim_data.tolist(),
    "dim": highdim_dim,
    "k": highdim_k,
    "nepochs": 5,
    "seed": 42,
    "finalLoss": float(highdim_model.obj[-1]),
    "lossPerEpoch": highdim_model.obj.tolist(),
}

with open("test/fixtures/emeans-faiss-reference-highdim.json", "w") as f:
    json.dump(highdim_fixture, f, indent=2)

print("Wrote test/fixtures/emeans-faiss-reference-highdim.json")
print("High-dim loss per epoch:", highdim_fixture["lossPerEpoch"])
print("High-dim cluster usage counts:", np.bincount(highdim_labels, minlength=highdim_k))

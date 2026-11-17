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

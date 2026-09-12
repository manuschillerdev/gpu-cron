"""MLX bidirectional affine-scan tagger with deployment-matched six-bit QAT."""

import argparse
import base64
import hashlib
import json
import time
from functools import partial
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

from data import (
    DATA,
    EMBEDDING_ROWS,
    FAMILIES,
    MAX_TOKENS,
    ROLES,
    SEED,
    VERSION,
    WIDTH,
    features,
    load_datasets,
    row,
    write_datasets,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "training/candidate"
FAMILY_COUNT = len(FAMILIES)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fake_quant(x):
    scale = mx.stop_gradient(mx.maximum(mx.max(mx.abs(x)) / 31, 1e-8))
    q = mx.clip(mx.round(x / scale), -31, 31) * scale
    return x + mx.stop_gradient(q - x)


def scan(a, b):
    # Parallel composition: (a2,b2) o (a1,b1) = (a2*a1, b2+a2*b1).
    offset = 1
    while offset < a.shape[1]:
        b = mx.concatenate(
            [b[:, :offset], b[:, offset:] + a[:, offset:] * b[:, :-offset]], axis=1
        )
        a = mx.concatenate([a[:, :offset], a[:, offset:] * a[:, :-offset]], axis=1)
        offset *= 2
    return b


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(EMBEDDING_ROWS, WIDTH)
        self.affine = nn.Linear(WIDTH, WIDTH * 2)
        self.hidden = nn.Linear(WIDTH * 3, 32)
        self.output = nn.Linear(32, len(FAMILIES) + len(ROLES))
        self.qat = False

    def token_scores(self, ids):
        def weight(x):
            return fake_quant(x) if self.qat else x

        def linear(layer, x):
            return x @ weight(layer.weight).T + weight(layer.bias)

        mask = (ids != 0)[..., None]
        bits = mx.maximum(ids.astype(mx.int32) - 1, 0)
        table = weight(self.embedding.weight)
        e = (
            (
                table[bits & 1023]
                + table[1024 + ((bits >> 10) & 255)]
                + table[1280 + ((bits >> 18) & 31)]
            )
            * (1 / 3**0.5)
            * mask
        )
        if self.training:
            e = e * mx.random.bernoulli(0.92, (*ids.shape, 1)) / 0.92
        ab = linear(self.affine, e)
        a = mx.where(mask, mx.sigmoid(ab[..., :WIDTH]), 1)
        b = mx.where(mask, mx.tanh(ab[..., WIDTH:]), 0)
        forward = scan(a, b)
        backward = scan(a[:, ::-1], b[:, ::-1])[:, ::-1]
        context = mx.concatenate([forward, backward], axis=-1)
        if self.training:
            context = context * mx.random.bernoulli(0.9, (*ids.shape, 1)) / 0.9
        hidden = nn.relu(linear(self.hidden, mx.concatenate([e, context], axis=-1)))
        scores = linear(self.output, hidden) * mask
        return scores

    def __call__(self, ids):
        scores = self.token_scores(ids)
        family = family_scores(scores, ids)
        return mx.concatenate(
            [family, scores[..., FAMILY_COUNT:].reshape(ids.shape[0], -1)], axis=1
        )


def family_scores(scores, ids):
    """Average family logits over real tokens, excluding padding."""
    count = mx.sum(ids != 0, axis=1, keepdims=True)
    return mx.sum(scores[..., :FAMILY_COUNT], axis=1) / mx.maximum(count, 1)


def training_step(model, optimizer):
    def loss_fn(net, x, y, roles):
        scores = net.token_scores(x)
        mask = x != 0
        family = family_scores(scores, x)
        family_loss = nn.losses.cross_entropy(family, y, reduction="mean")
        tag_loss = nn.losses.cross_entropy(scores[..., FAMILY_COUNT:], roles)
        return family_loss + mx.sum(tag_loss * mask) / mx.maximum(mx.sum(mask), 1)

    grad = nn.value_and_grad(model, loss_fn)
    optimizer.init(model.trainable_parameters())
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y, roles):
        loss, grads = grad(model, x, y, roles)
        grads, _ = optim.clip_grad_norm(grads, 1)
        optimizer.update(model, grads)
        return loss

    return step, state


def use_gpu():
    if not mx.metal.is_available():
        raise RuntimeError("Cron training requires an Apple Metal GPU.")
    mx.set_default_device(mx.gpu)
    mx.random.seed(SEED)


def arrays(rows):
    # Pad to a power of two so every parallel scan has the same shape.
    length = 2 ** (max(len(r[2]) for r in rows) - 1).bit_length()
    return (
        mx.array(np.stack([features(t, length) for t, _, _ in rows])),
        mx.array([f for _, f, _ in rows], dtype=mx.int32),
        mx.array([r + [0] * (length - len(r)) for _, _, r in rows], dtype=mx.int32),
    )


def pack_weights(model):
    segments, values = [], []
    for name, value in tree_flatten(model.parameters()):
        arr = np.asarray(value)
        scale = float(max(np.abs(arr).max() / 31, 1e-8))
        q = np.clip(np.round(arr / scale), -31, 31).astype(np.int32)
        segments.append(
            {
                "name": name,
                "offset": len(values),
                "length": q.size,
                "shape": list(arr.shape),
                "scale": scale,
            }
        )
        values.extend(q.flatten().tolist())
        model.update(
            tree_unflatten(
                [(name, mx.array((q.astype(np.float64) * scale).astype(np.float32)))]
            )
        )
    model.qat = False
    packed = bytearray()
    bits = available = 0
    for q in values:
        bits |= (q & 63) << available
        available += 6
        while available >= 8:
            packed.append(bits & 255)
            bits >>= 8
            available -= 8
    if available:
        packed.append(bits)
    payload = {
        "format": 3,
        "architecture": "feature-sum-bidirectional-affine-scan",
        "features": MAX_TOKENS,
        "embeddingRows": EMBEDDING_ROWS,
        "hidden": WIDTH,
        "families": FAMILIES,
        "roles": ROLES,
        "parameters": len(values),
        "segments": segments,
        "data": base64.b64encode(packed).decode(),
    }
    artifact = json.dumps(payload, separators=(",", ":")) + "\n"
    return payload, artifact, len(packed)


def fit(sets, epochs):
    training = [row(item) for item in sets["train"]]
    x, y, r = arrays(training)
    # Equal family sampling prevents rare interval/unsupported meanings being swamped
    # by the much larger space of clock + weekday combinations.
    family_indices = [
        mx.array(
            [i for i, (_, label, _) in enumerate(training) if label == family],
            dtype=mx.int32,
        )
        for family in range(FAMILY_COUNT)
    ]
    if any(not len(ids) for ids in family_indices):
        raise ValueError("Training split must represent every family")
    model = Model()
    optimizer = optim.AdamW(
        learning_rate=0.003, weight_decay=0.001, bias_correction=True
    )
    model.qat = epochs <= 12
    step, state = training_step(model, optimizer)
    mx.eval(x, y, r, state)
    train_started = time.perf_counter()
    print(
        f"MLX {version('mlx')} / {mx.device_info()['device_name']}; {len(training)} sequences, length {x.shape[1]}",
        flush=True,
    )
    for epoch in range(epochs):
        if epoch == epochs - 12:
            model.qat = True
            step, state = training_step(
                model, optimizer
            )  # Retrace with fake quantization.
        lr = (
            0.003
            * min(1, (epoch + 1) / 3)
            * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * epoch / epochs)))
        )
        optimizer.learning_rate = lr
        per_family = (len(x) + FAMILY_COUNT - 1) // FAMILY_COUNT
        sampled = mx.concatenate(
            [
                ids[mx.random.randint(0, len(ids), (per_family,))]
                for ids in family_indices
            ]
        )
        order = sampled[mx.random.permutation(len(sampled))][: len(x)]
        for start in range(0, len(x), 256):
            ids = order[start : start + 256]
            loss = step(x[ids], y[ids], r[ids])
            mx.eval(state, loss)
        print(
            f"epoch {epoch + 1}/{epochs} loss={loss.item():.5f} qat={model.qat}",
            flush=True,
        )
    mx.synchronize()
    train_seconds = time.perf_counter() - train_started
    model.eval()
    return model, train_seconds, len(training)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA,
        help="Directory containing the annotated JSONL splits",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    started = time.perf_counter()
    sets = load_datasets(args.data_dir)
    CANDIDATE.mkdir(parents=True, exist_ok=True)
    use_gpu()
    # Evaluation must use the same records as this training run.
    write_datasets(sets, CANDIDATE / "data")
    model, train_seconds, training_examples = fit(sets, args.epochs)
    payload, artifact, packed_bytes = pack_weights(model)
    (CANDIDATE / "weights.json").write_text(artifact)
    # Export quantized MLX outputs for the browser parity check.
    fixture_path = ROOT / "test/model-fixtures.json"
    texts = [f["text"] for f in json.loads(fixture_path.read_text())]
    xx = mx.array(np.stack([features(t) for t in texts]))
    scores = model(xx).tolist()
    fixture_path = CANDIDATE / "model-fixtures.json"
    fixture_path.write_text(
        json.dumps(
            [
                {"text": t, "features": features(t).tolist(), "logits": s}
                for t, s in zip(texts, scores, strict=True)
            ],
            separators=(",", ":"),
        )
        + "\n"
    )
    report = {
        "seed": SEED,
        "framework": "mlx",
        "frameworkVersion": version("mlx"),
        "device": "gpu",
        "deviceName": mx.device_info()["device_name"],
        "architecture": payload["architecture"],
        "parameters": payload["parameters"],
        "packedWeightBytes": packed_bytes,
        "quantization": "signed six-bit per tensor, final 12 epochs deployment-matched fake quantization",
        "epochs": args.epochs,
        "batchSize": 256,
        "trainingExamples": training_examples,
        "trainingSeconds": train_seconds,
        "dataVersion": VERSION,
        "dataSourceSha256": sha(ROOT / "training/data.py"),
        "trainingSourceSha256": sha(Path(__file__)),
        "weightsSha256": sha(CANDIDATE / "weights.json"),
        "fixturesSha256": sha(fixture_path),
        "manifestSha256": sha(CANDIDATE / "data/manifest.json"),
        "totalSeconds": time.perf_counter() - started,
    }
    (CANDIDATE / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Candidate written to {CANDIDATE}. Run pnpm run evaluate:cron:candidate.")


if __name__ == "__main__":
    main()

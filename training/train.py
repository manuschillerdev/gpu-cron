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
BATCH_SIZE = 256
LEARNING_RATE = 0.003
QUANTIZED_EPOCHS = 12


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fake_quant(tensor):
    """Round to signed six-bit values while keeping gradients for training."""
    scale = mx.stop_gradient(mx.maximum(mx.max(mx.abs(tensor)) / 31, 1e-8))
    quantized = mx.clip(mx.round(tensor / scale), -31, 31) * scale
    return tensor + mx.stop_gradient(quantized - tensor)


def affine_scan(gates, candidates):
    """Compute h = gate * previous_h + candidate for every token in parallel."""
    # Composition: (a2,b2) after (a1,b1) = (a2*a1, b2+a2*b1).
    offset = 1
    while offset < gates.shape[1]:
        candidates = mx.concatenate(
            [
                candidates[:, :offset],
                candidates[:, offset:] + gates[:, offset:] * candidates[:, :-offset],
            ],
            axis=1,
        )
        gates = mx.concatenate(
            [gates[:, :offset], gates[:, offset:] * gates[:, :-offset]],
            axis=1,
        )
        offset *= 2
    return candidates


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(EMBEDDING_ROWS, WIDTH)
        self.affine = nn.Linear(WIDTH, WIDTH * 2)
        self.hidden = nn.Linear(WIDTH * 3, 32)
        self.output = nn.Linear(32, len(FAMILIES) + len(ROLES))
        self.qat = False

    def token_scores(self, token_ids):
        def deployment_weight(tensor):
            return fake_quant(tensor) if self.qat else tensor

        def linear(layer, inputs):
            return inputs @ deployment_weight(layer.weight).T + deployment_weight(
                layer.bias
            )

        real_tokens = (token_ids != 0)[..., None]
        feature_bits = mx.maximum(token_ids.astype(mx.int32) - 1, 0)
        table = deployment_weight(self.embedding.weight)
        embeddings = (
            (
                table[feature_bits & 1023]
                + table[1024 + ((feature_bits >> 10) & 255)]
                + table[1280 + ((feature_bits >> 18) & 31)]
            )
            * (1 / 3**0.5)
            * real_tokens
        )
        if self.training:
            embeddings = (
                embeddings * mx.random.bernoulli(0.92, (*token_ids.shape, 1)) / 0.92
            )

        recurrence = linear(self.affine, embeddings)
        gates = mx.where(real_tokens, mx.sigmoid(recurrence[..., :WIDTH]), 1)
        candidates = mx.where(real_tokens, mx.tanh(recurrence[..., WIDTH:]), 0)
        forward_context = affine_scan(gates, candidates)
        backward_context = affine_scan(gates[:, ::-1], candidates[:, ::-1])[:, ::-1]
        context = mx.concatenate([forward_context, backward_context], axis=-1)
        if self.training:
            context = context * mx.random.bernoulli(0.9, (*token_ids.shape, 1)) / 0.9

        hidden = nn.relu(
            linear(self.hidden, mx.concatenate([embeddings, context], axis=-1))
        )
        return linear(self.output, hidden) * real_tokens

    def __call__(self, token_ids):
        scores = self.token_scores(token_ids)
        family = family_scores(scores, token_ids)
        return mx.concatenate(
            [family, scores[..., FAMILY_COUNT:].reshape(token_ids.shape[0], -1)],
            axis=1,
        )


def family_scores(scores, token_ids):
    """Average family logits over real tokens, excluding padding."""
    count = mx.sum(token_ids != 0, axis=1, keepdims=True)
    return mx.sum(scores[..., :FAMILY_COUNT], axis=1) / mx.maximum(count, 1)


def training_step(model, optimizer):
    def loss_fn(net, token_ids, family_labels, role_labels):
        scores = net.token_scores(token_ids)
        real_tokens = token_ids != 0
        families = family_scores(scores, token_ids)
        family_loss = nn.losses.cross_entropy(families, family_labels, reduction="mean")
        role_loss = nn.losses.cross_entropy(scores[..., FAMILY_COUNT:], role_labels)
        mean_role_loss = mx.sum(role_loss * real_tokens) / mx.maximum(
            mx.sum(real_tokens), 1
        )
        return family_loss + mean_role_loss

    grad = nn.value_and_grad(model, loss_fn)
    optimizer.init(model.trainable_parameters())
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(token_ids, family_labels, role_labels):
        loss, grads = grad(model, token_ids, family_labels, role_labels)
        grads, _ = optim.clip_grad_norm(grads, 1)
        optimizer.update(model, grads)
        return loss

    return step, state


def use_gpu():
    if not mx.metal.is_available():
        raise RuntimeError("Cron training requires an Apple Metal GPU.")
    mx.set_default_device(mx.gpu)
    mx.random.seed(SEED)


def make_batch(rows):
    # Pad to a power of two so every parallel scan has the same shape.
    longest_sequence = max(len(role_labels) for _, _, role_labels in rows)
    length = 2 ** (longest_sequence - 1).bit_length()
    return (
        mx.array(np.stack([features(text, length) for text, _, _ in rows])),
        mx.array([family for _, family, _ in rows], dtype=mx.int32),
        mx.array(
            [roles + [0] * (length - len(roles)) for _, _, roles in rows],
            dtype=mx.int32,
        ),
    )


def pack_weights(model):
    segments = []
    quantized_values = []
    for name, value in tree_flatten(model.parameters()):
        tensor = np.asarray(value)
        scale = float(max(np.abs(tensor).max() / 31, 1e-8))
        quantized = np.clip(np.round(tensor / scale), -31, 31).astype(np.int32)
        segments.append(
            {
                "name": name,
                "offset": len(quantized_values),
                "length": quantized.size,
                "shape": list(tensor.shape),
                "scale": scale,
            }
        )
        quantized_values.extend(quantized.flatten().tolist())
        model.update(
            tree_unflatten(
                [
                    (
                        name,
                        mx.array(
                            (quantized.astype(np.float64) * scale).astype(np.float32)
                        ),
                    )
                ]
            )
        )
    model.qat = False
    packed = bytearray()
    bit_buffer = 0
    bit_count = 0
    for value in quantized_values:
        bit_buffer |= (value & 63) << bit_count
        bit_count += 6
        while bit_count >= 8:
            packed.append(bit_buffer & 255)
            bit_buffer >>= 8
            bit_count -= 8
    if bit_count:
        packed.append(bit_buffer)
    payload = {
        "format": 3,
        "architecture": "feature-sum-bidirectional-affine-scan",
        "features": MAX_TOKENS,
        "embeddingRows": EMBEDDING_ROWS,
        "hidden": WIDTH,
        "families": FAMILIES,
        "roles": ROLES,
        "parameters": len(quantized_values),
        "segments": segments,
        "data": base64.b64encode(packed).decode(),
    }
    artifact = json.dumps(payload, separators=(",", ":")) + "\n"
    return payload, artifact, len(packed)


def fit(sets, epochs):
    training_rows = [row(item) for item in sets["train"]]
    token_ids, family_labels, role_labels = make_batch(training_rows)
    # Equal family sampling prevents rare interval/unsupported meanings being swamped
    # by the much larger space of clock + weekday combinations.
    family_indices = [
        mx.array(
            [i for i, (_, label, _) in enumerate(training_rows) if label == family],
            dtype=mx.int32,
        )
        for family in range(FAMILY_COUNT)
    ]
    if any(not len(ids) for ids in family_indices):
        raise ValueError("Training split must represent every family")
    model = Model()
    optimizer = optim.AdamW(
        learning_rate=LEARNING_RATE, weight_decay=0.001, bias_correction=True
    )
    model.qat = epochs <= QUANTIZED_EPOCHS
    step, state = training_step(model, optimizer)
    mx.eval(token_ids, family_labels, role_labels, state)
    train_started = time.perf_counter()
    print(
        f"MLX {version('mlx')} / {mx.device_info()['device_name']}; "
        f"{len(training_rows)} sequences, length {token_ids.shape[1]}",
        flush=True,
    )
    for epoch in range(epochs):
        if epoch == epochs - QUANTIZED_EPOCHS:
            model.qat = True
            step, state = training_step(
                model, optimizer
            )  # Retrace with fake quantization.
        lr = (
            LEARNING_RATE
            * min(1, (epoch + 1) / 3)
            * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * epoch / epochs)))
        )
        optimizer.learning_rate = lr
        per_family = (len(token_ids) + FAMILY_COUNT - 1) // FAMILY_COUNT
        sampled = mx.concatenate(
            [
                ids[mx.random.randint(0, len(ids), (per_family,))]
                for ids in family_indices
            ]
        )
        order = sampled[mx.random.permutation(len(sampled))][: len(token_ids)]
        for start in range(0, len(token_ids), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            loss = step(token_ids[batch], family_labels[batch], role_labels[batch])
            mx.eval(state, loss)
        print(
            f"epoch {epoch + 1}/{epochs} loss={loss.item():.5f} qat={model.qat}",
            flush=True,
        )
    mx.synchronize()
    train_seconds = time.perf_counter() - train_started
    model.eval()
    return model, train_seconds, len(training_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
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
    token_ids = mx.array(np.stack([features(text) for text in texts]))
    logits = model(token_ids).tolist()
    fixture_path = CANDIDATE / "model-fixtures.json"
    fixture_path.write_text(
        json.dumps(
            [
                {"text": text, "features": features(text).tolist(), "logits": scores}
                for text, scores in zip(texts, logits, strict=True)
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
        "batchSize": BATCH_SIZE,
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

"""Measure synchronized MLX Metal training epochs; run through mise and uv."""
import json
import statistics
import time
from importlib.metadata import version

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from train import ROOT, SEED, Model, corpus, arrays, training_step, use_gpu


def main():
    use_gpu()
    rows = corpus(SEED, 21000)
    x, y, roles = arrays(rows)
    mx.eval(x, y, roles)
    results = []
    for batch in [256, 1024]:
        mx.random.seed(SEED)
        model = Model()
        optimizer = optim.AdamW(learning_rate=.008, weight_decay=.001, bias_correction=True)
        step, state = training_step(model, optimizer)

        def epoch():
            order = mx.random.permutation(len(x))
            for start in range(0, len(x), batch):
                ids = order[start:start + batch]
                loss = step(x[ids], y[ids], roles[ids])
                mx.eval(state, loss)
            mx.synchronize()

        epoch()  # Warm up compilation/allocation for full and partial batches.
        elapsed = []
        for _ in range(3):
            started = time.perf_counter()
            epoch()
            elapsed.append(time.perf_counter() - started)
        seconds = statistics.median(elapsed)
        results.append({'device': 'gpu', 'batch': batch, 'medianEpochSeconds': seconds, 'examplesPerSecond': round(len(rows) / seconds)})
    report = {'framework': 'mlx', 'frameworkVersion': version('mlx'), 'deviceInfo': mx.device_info(), 'examples': len(rows), 'warmupEpochs': 1, 'measuredEpochs': 3, 'compiled': True, 'synchronized': True, 'results': results, 'notes': 'Forward/backward/AdamW and GPU shuffle with resident data. Excludes feature generation, export and validation. Batch-size throughput is not convergence-equivalent. Does not compare frameworks or change the training configuration.'}
    (ROOT / 'training/hardware-benchmark.local.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

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
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np
from data import SEED, VOCAB, WIDTH, MAX_TOKENS, FAMILIES, ROLES, VERSION, corpus, features, tokens, datasets, write_manifest, write_vocabulary, row

ROOT = Path(__file__).resolve().parents[1]

def fake_quant(x):
    scale = mx.stop_gradient(mx.maximum(mx.max(mx.abs(x)) / 31, 1e-8))
    q = mx.clip(mx.round(x / scale), -31, 31) * scale
    return x + mx.stop_gradient(q - x)

def scan(a, b):
    # Parallel composition: (a2,b2) o (a1,b1) = (a2*a1, b2+a2*b1).
    offset = 1
    while offset < a.shape[1]:
        b = mx.concatenate([b[:, :offset], b[:, offset:] + a[:, offset:] * b[:, :-offset]], axis=1)
        a = mx.concatenate([a[:, :offset], a[:, offset:] * a[:, :-offset]], axis=1)
        offset *= 2
    return b

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, WIDTH)
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
        e = (table[bits & 1023] + table[1024 + ((bits >> 10) & 255)] + table[1280 + ((bits >> 18) & 31)]) * (1 / 3**.5) * mask
        if self.training:
            e = e * mx.random.bernoulli(.92, (*ids.shape,1)) / .92
        ab = linear(self.affine, e)
        a = mx.where(mask, mx.sigmoid(ab[..., :WIDTH]), 1)
        b = mx.where(mask, mx.tanh(ab[..., WIDTH:]), 0)
        forward = scan(a, b)
        backward = scan(a[:, ::-1], b[:, ::-1])[:, ::-1]
        context = mx.concatenate([forward, backward], axis=-1)
        if self.training:
            context = context * mx.random.bernoulli(.9, (*ids.shape,1)) / .9
        hidden = nn.relu(linear(self.hidden, mx.concatenate([e, context], axis=-1)))
        scores = linear(self.output, hidden) * mask
        return scores

    def __call__(self, ids):
        scores = self.token_scores(ids)
        mask = (ids != 0)[..., None]
        family = mx.sum(scores[..., :7], axis=1) / mx.maximum(mx.sum(mask, axis=1), 1)
        return mx.concatenate([family, scores[..., 7:].reshape(ids.shape[0], -1)], axis=1)

def training_step(model, optimizer):
    def loss_fn(net, x, y, roles):
        scores = net.token_scores(x)
        mask = x != 0
        family = mx.sum(scores[..., :7],axis=1)/mx.maximum(mx.sum(mask,axis=1,keepdims=True),1)
        family_loss = nn.losses.cross_entropy(family, y, reduction='mean')
        tag_loss = nn.losses.cross_entropy(scores[..., 7:], roles)
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
        raise RuntimeError('Cron training requires an Apple Metal GPU.')
    mx.set_default_device(mx.gpu)
    mx.random.seed(SEED)

def arrays(rows, length=None):
    length = length or 2 ** (max(len(r[2]) for r in rows) - 1).bit_length()
    return (mx.array(np.stack([features(t, length) for t,_,_ in rows])),
            mx.array([f for _,f,_ in rows], dtype=mx.int32),
            mx.array([r + [0] * (length-len(r)) for _,_,r in rows], dtype=mx.int32))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=45)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    use_gpu()
    started = time.perf_counter()
    sets = datasets()
    manifest = write_manifest(sets)  # Commit split identity before fitting any weights.
    vocabulary_size = write_vocabulary(sets['train'])
    training = [row(item) for item in sets['train']]
    validation = [row(item) for item in sets['development']]
    x,y,r = arrays(training,64)
    # Equal family sampling prevents rare interval/unsupported meanings being swamped
    # by the much larger space of clock + weekday combinations.
    family_indices = [mx.array([i for i,(_,label,_) in enumerate(training) if label==family],dtype=mx.int32) for family in range(7)]
    if any(not len(ids) for ids in family_indices):
        raise ValueError('Training split must represent every family')
    model = Model()
    checkpoint = ROOT / 'training/checkpoint.safetensors'
    state_path = ROOT / 'training/optimizer.safetensors'
    progress_path = ROOT / 'training/progress.json'
    optimizer = optim.AdamW(learning_rate=.003, weight_decay=.001, bias_correction=True)
    start_epoch = 0
    if args.resume:
        progress = json.loads(progress_path.read_text())
        if progress['epochs'] != args.epochs or progress['dataVersion'] != VERSION or progress.get('splitIdentity') != manifest['splits']:
            raise ValueError('Resume requires the original epochs and dataset version')
        model.load_weights(str(checkpoint))
        optimizer.state = tree_unflatten(list(mx.load(str(state_path)).items()))
        mx.random.state = [mx.array(v, dtype=mx.uint32) for v in progress['randomState']]
        start_epoch = progress['epoch'] + 1
    if start_epoch:
        training = [row(item) for item in datasets(start_epoch)['train']]
        x,y,r = arrays(training,64)
        family_indices = [mx.array([i for i,(_,label,_) in enumerate(training) if label==family],dtype=mx.int32) for family in range(7)]
    model.qat = start_epoch >= args.epochs - 12
    step,state = training_step(model, optimizer)
    mx.eval(x,y,r,state)
    train_started = time.perf_counter()
    print(f'MLX {version("mlx")} / {mx.device_info()["device_name"]}; {len(training)} sequences, length {x.shape[1]}', flush=True)
    for epoch in range(start_epoch, args.epochs):
        if epoch > start_epoch:
            training = [row(item) for item in datasets(epoch)['train']]
            x,y,r = arrays(training,64)
            family_indices = [mx.array([i for i,(_,label,_) in enumerate(training) if label==family],dtype=mx.int32) for family in range(7)]
        if epoch == args.epochs - 12:
            model.qat = True
            step,state = training_step(model, optimizer)  # Retrace with fake quantization.
        lr = .003 * min(1, (epoch+1)/3) * (.1 + .9*.5*(1+np.cos(np.pi*epoch/args.epochs)))
        optimizer.learning_rate = lr
        per_family = (len(x)+6)//7
        sampled = mx.concatenate([ids[mx.random.randint(0,len(ids),(per_family,))] for ids in family_indices])
        order = sampled[mx.random.permutation(len(sampled))][:len(x)]
        for start in range(0, len(x), 256):
            ids = order[start:start+256]
            loss = step(x[ids], y[ids], r[ids])
            mx.eval(state, loss)
        model.save_weights(str(checkpoint))
        mx.save_safetensors(str(state_path), dict(tree_flatten(optimizer.state)))
        progress_path.write_text(json.dumps({'epoch':epoch,'epochs':args.epochs,'dataVersion':VERSION,'splitIdentity':manifest['splits'],'randomState':[a.tolist() for a in mx.random.state]}))
        print(f'epoch {epoch+1}/{args.epochs} loss={loss.item():.5f} qat={model.qat}', flush=True)
    mx.synchronize()
    train_seconds = time.perf_counter()-train_started
    model.eval()
    segments, values = [], []
    for name, value in tree_flatten(model.parameters()):
        arr = np.asarray(value)
        scale = float(max(np.abs(arr).max()/31, 1e-8))
        q = np.clip(np.round(arr/scale), -31,31).astype(np.int32)
        segments.append({'name':name,'offset':len(values),'length':q.size,'shape':list(arr.shape),'scale':scale})
        values.extend(q.flatten().tolist())
        model.update(tree_unflatten([(name,mx.array((q.astype(np.float64)*scale).astype(np.float32)))]))
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
    payload = {'format':3,'architecture':'feature-sum-bidirectional-affine-scan','features':MAX_TOKENS,'vocabulary':VOCAB,'hidden':WIDTH,'families':FAMILIES,'roles':ROLES,'parameters':len(values),'segments':segments,'data':base64.b64encode(packed).decode()}
    artifact = json.dumps(payload,separators=(',',':'))+'\n'
    (ROOT/'src/model/weights.json').write_text(artifact)
    def evaluate(rows):
        correct = exact = token_correct = token_count = 0
        for start in range(0,len(rows),256):
            xx,yy,rr = arrays(rows[start:start+256])
            out = model(xx)
            family = out[:,:7].argmax(1)
            tags = out[:,7:].reshape(*xx.shape,len(ROLES)).argmax(-1)
            ok = (tags == rr) | (xx == 0)
            correct += int(mx.sum(family == yy).item())
            exact += int(mx.sum(mx.all(ok,axis=1) & (family == yy)).item())
            token_correct += int(mx.sum((tags == rr) & (xx != 0)).item())
            token_count += int(mx.sum(xx != 0).item())
        return {'correct':correct,'total':len(rows),'familyAccuracy':correct/len(rows),'exactFamilyAndSpans':exact/len(rows),'tokenAccuracy':token_correct/token_count}
    # Reuse the existing verification examples; do not add test cases.
    fixture_path = ROOT/'test/model-fixtures.json'
    texts = [f['text'] for f in json.loads(fixture_path.read_text())]
    xx = mx.array(np.stack([features(t) for t in texts]))
    scores = model(xx).tolist()
    fixture_path.write_text(json.dumps([{'text':t,'features':features(t).tolist(),'logits':s} for t,s in zip(texts,scores)],separators=(',',':'))+'\n')
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = {'seed':SEED,'framework':'mlx','frameworkVersion':version('mlx'),'device':'gpu','deviceInfo':mx.device_info(),'compiled':True,'architecture':payload['architecture'],'parameters':len(values),'packedWeightBytes':len(packed),'quantization':'signed six-bit per tensor, final 12 epochs deployment-matched fake quantization','epochs':args.epochs,'batchSize':256,'sampling':'Fresh phrasings every epoch; uniform family sampling from fixed training-only meaning groups','regularization':'8% token embedding dropout and 10% bidirectional context dropout during training only','trainingExamples':len(training),'trainingSeconds':train_seconds,'dataVersion':VERSION,'vocabularySize':vocabulary_size,'vocabularySha256':sha(ROOT/'src/model/vocabulary.json'),'dataSourceSha256':sha(ROOT/'training/data.py'),'trainingSourceSha256':sha(Path(__file__)),'weightsSha256':sha(ROOT/'src/model/weights.json'),'fixturesSha256':sha(fixture_path),'validation':evaluate(validation),'holdouts':{'status':'not evaluated by trainer','command':'pnpm run evaluate:cron:holdout','splits':manifest['splits']},'split':manifest['grouping'],'manifestSha256':sha(ROOT/'training/data/manifest.json'),'limitation':'Family plus token-span metrics are not end-to-end schedule accuracy. The typed compiler still interprets values and checks schedule semantics. Real-user accuracy is unmeasured.','totalSeconds':time.perf_counter()-started}
    (ROOT/'training/report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()

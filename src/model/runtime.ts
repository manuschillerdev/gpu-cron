import { decodeWeights, SEGMENTS, OUTPUT_COUNT } from './parameters.js';
import { FEATURE_COUNT, HIDDEN_COUNT, FAMILIES, ROLES } from '../features.js';

const H = HIDDEN_COUNT;
const R = ROLES.length;
const F = FAMILIES.length;
const SCAN_LEVELS = Math.log2(FEATURE_COUNT);
const PROJECT = 1 + SCAN_LEVELS;
const TOKEN_ROLES = PROJECT + 1;
const FAMILY_SCORES = PROJECT + 2;
const offset = (name: string) => {
  const segment = SEGMENTS.find((s) => s.name === name);
  if (!segment) throw new Error(`Missing model tensor ${name}`);
  return segment.offset;
};
/** The five model operations below mirror Model.token_scores in training/train.py. */
function shaders(): string[] {
  const embeddingWeight = offset('embedding.weight');
  const affineWeight = offset('affine.weight');
  const affineBias = offset('affine.bias');
  const hiddenWeight = offset('hidden.weight');
  const hiddenBias = offset('hidden.bias');
  const outputWeight = offset('output.weight');
  const outputBias = offset('output.bias');

  const common = `
struct Batch { rows: u32, length: u32 }
@group(0) @binding(0) var<storage, read> ids: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> source: array<f32>;
@group(0) @binding(3) var<storage, read_write> destination: array<f32>;
@group(0) @binding(4) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<uniform> batch: Batch;

fn embedding(id: u32, channel: u32) -> f32 {
  if (id == 0u) { return 0.0; }
  let bits = id - 1u;
  let word = bits & 1023u;
  let consonants = 1024u + ((bits >> 10u) & 255u);
  let shape = 1280u + ((bits >> 18u) & 31u);
  return (
    weights[${embeddingWeight}u + word * ${H}u + channel] +
    weights[${embeddingWeight}u + consonants * ${H}u + channel] +
    weights[${embeddingWeight}u + shape * ${H}u + channel]
  ) * 0.5773502691896258;
}

fn token(index: u32) -> u32 {
  let row = index / batch.length;
  let position = index % batch.length;
  return u32(ids[row * ${FEATURE_COUNT}u + position]);
}
`;
  const kernel = (body: string) => `${common}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) thread: vec3<u32>) {
${body}
}
`;

  // Each token/channel starts with a gate a and candidate b for h = a*h_previous + b.
  // Store the same (a, b) pair for the forward and backward scans.
  const embed = kernel(`
  let index = thread.x / ${H}u;
  let channel = thread.x % ${H}u;
  if (index >= batch.rows * batch.length) { return; }
  let id = token(index);
  var a = 1.0;
  var b = 0.0;
  if (id != 0u) {
    var gate = weights[${affineBias}u + channel];
    var candidate = weights[${affineBias + H}u + channel];
    for (var k = 0u; k < ${H}u; k++) {
      let value = embedding(id, k);
      gate += value * weights[${affineWeight}u + channel * ${H}u + k];
      candidate += value * weights[${affineWeight + H * H}u + channel * ${H}u + k];
    }
    a = 1.0 / (1.0 + exp(-gate));
    b = tanh(candidate);
  }
  let base = index * ${4 * H}u + channel * 2u;
  destination[base] = a;
  destination[base + 1u] = b;
  destination[base + ${2 * H}u] = a;
  destination[base + ${2 * H + 1}u] = b;
`);

  // Compose pairs at distances 1, 2, 4, ... . Padding is the identity pair (1, 0).
  const scans = Array.from({ length: SCAN_LEVELS }, (_, level) =>
    kernel(`
  let index = thread.x / ${2 * H}u;
  let channel = thread.x % ${2 * H}u;
  if (index >= batch.rows * batch.length) { return; }
  let position = index % batch.length;
  let base = index * ${4 * H}u + channel * 2u;
  var a = source[base];
  var b = source[base + 1u];
  let forward = channel < ${H}u;
  let distance = ${2 ** level}u;
  let valid = select((position + distance < batch.length), (position >= distance), forward);
  if (valid) {
    let neighbor = select(index + distance, index - distance, forward);
    let other = neighbor * ${4 * H}u + channel * 2u;
    b += a * source[other + 1u];
    a *= source[other];
  }
  destination[base] = a;
  destination[base + 1u] = b;
`),
  );

  // Project the embedding and both scan contexts into 32 hidden channels, then ReLU.
  const project = kernel(`
  let index = thread.x / 32u;
  let channel = thread.x % 32u;
  if (index >= batch.rows * batch.length) { return; }
  let id = token(index);
  var sum = weights[${hiddenBias}u + channel];
  for (var k = 0u; k < ${H}u; k++) {
    sum += embedding(id, k) * weights[${hiddenWeight}u + channel * ${3 * H}u + k];
    sum += source[index * ${4 * H}u + k * 2u + 1u]
      * weights[${hiddenWeight + H}u + channel * ${3 * H}u + k];
    sum += source[index * ${4 * H}u + ${2 * H}u + k * 2u + 1u]
      * weights[${hiddenWeight + 2 * H}u + channel * ${3 * H}u + k];
  }
  hidden[index * 32u + channel] = max(0.0, sum);
`);

  const roles = kernel(`
  let index = thread.x / ${R}u;
  let channel = thread.x % ${R}u;
  if (index >= batch.rows * batch.length) { return; }
  var sum = weights[${outputBias + F}u + channel];
  for (var k = 0u; k < 32u; k++) {
    sum += hidden[index * 32u + k] * weights[${outputWeight + F * 32}u + channel * 32u + k];
  }
  let row = index / batch.length;
  let position = index % batch.length;
  output[row * ${OUTPUT_COUNT}u + ${F}u + position * ${R}u + channel]
    = select(sum, 0.0, token(index) == 0u);
`);

  // A schedule's family scores are the mean of its non-padding token scores.
  const family = kernel(`
  let row = thread.x / ${F}u;
  let channel = thread.x % ${F}u;
  if (row >= batch.rows) { return; }
  var total = 0.0;
  var count = 0.0;
  for (var position = 0u; position < batch.length; position++) {
    let index = row * batch.length + position;
    if (token(index) == 0u) { continue; }
    var sum = weights[${outputBias}u + channel];
    for (var k = 0u; k < 32u; k++) {
      sum += hidden[index * 32u + k] * weights[${outputWeight}u + channel * 32u + k];
    }
    total += sum;
    count += 1.0;
  }
  output[row * ${OUTPUT_COUNT}u + channel] = total / max(count, 1.0);
`);
  return [embed, ...scans, project, roles, family];
}

export class CronModel {
  outputLocations: string[] = [];
  readonly executionProvider = 'webgpu' as const;
  private closed = false;
  private pending: Promise<unknown> = Promise.resolve();
  private lost?: string;
  private buffers: GPUBuffer[] = [];
  private groups: GPUBindGroup[] = [];
  private readback?: GPUBuffer;
  private capacity = 0;
  private constructor(
    private device: GPUDevice,
    private weights: GPUBuffer,
    private uniform: GPUBuffer,
    private layout: GPUBindGroupLayout,
    private pipelines: GPUComputePipeline[],
  ) {
    void device.lost.then((info) => {
      this.lost = info.message || 'WebGPU device lost';
    });
  }
  static async create(): Promise<CronModel> {
    const adapter = await globalThis.navigator?.gpu?.requestAdapter();
    if (!adapter)
      throw new Error('WebGPU is required. Use a browser and device with WebGPU support.');
    const device = await adapter.requestDevice();
    try {
      const layout = device.createBindGroupLayout({
        entries: Array.from({ length: 7 }, (_, binding) => ({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: binding === 6 ? 'uniform' : binding < 3 ? 'read-only-storage' : 'storage',
          },
        })),
      });
      const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [layout],
      });
      const pipelines = await Promise.all(
        shaders().map((code) =>
          device.createComputePipelineAsync({
            layout: pipelineLayout,
            compute: {
              module: device.createShaderModule({ code }),
              entryPoint: 'main',
            },
          }),
        ),
      );
      const values = decodeWeights();
      const weights = device.createBuffer({
        size: values.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(weights, 0, values as Float32Array<ArrayBuffer>);
      const uniform = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      return new CronModel(device, weights, uniform, layout, pipelines);
    } catch (error) {
      device.destroy();
      throw error;
    }
  }
  private allocate(rows: number): void {
    if (rows <= this.capacity) return;
    const capacity = 2 ** Math.ceil(Math.log2(rows));
    const widths = [
      FEATURE_COUNT,
      FEATURE_COUNT * 4 * H,
      FEATURE_COUNT * 4 * H,
      FEATURE_COUNT * 32,
      OUTPUT_COUNT,
    ];
    if (widths.some((w) => capacity * w * 4 > this.device.limits.maxStorageBufferBindingSize))
      throw new RangeError('Batch exceeds WebGPU limits');
    this.buffers.forEach((b) => b.destroy());
    this.readback?.destroy();
    this.buffers = widths.map((w) =>
      this.device.createBuffer({
        size: capacity * w * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }),
    );
    this.readback = this.device.createBuffer({
      size: capacity * OUTPUT_COUNT * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.groups = [0, 1].map((swap) =>
      this.device.createBindGroup({
        layout: this.layout,
        entries: [
          this.buffers[0]!,
          this.weights,
          this.buffers[1 + swap]!,
          this.buffers[2 - swap]!,
          this.buffers[3]!,
          this.buffers[4]!,
          this.uniform,
        ].map((buffer, binding) => ({ binding, resource: { buffer } })),
      }),
    );
    this.capacity = capacity;
  }
  logits(input: Float32Array): Promise<Float32Array> {
    if (this.closed) return Promise.reject(new Error('Model is disposed.'));
    if (input.length % FEATURE_COUNT)
      return Promise.reject(new Error('Invalid model input shape.'));
    const snapshot = new Float32Array(input);
    const result = this.pending.then(async () => {
      if (this.lost) throw new Error(this.lost);
      const rows = snapshot.length / FEATURE_COUNT;
      if (!rows) return new Float32Array();
      const output = new Float32Array(rows * OUTPUT_COUNT);
      // Bound storage use for the public 4096-expression batch limit.
      for (let start = 0; start < rows; start += 128) {
        const count = Math.min(128, rows - start),
          data = snapshot.subarray(start * FEATURE_COUNT, (start + count) * FEATURE_COUNT);
        let length = 1;
        for (let i = 0; i < data.length; i++)
          if (data[i] !== 0) length = Math.max(length, (i % FEATURE_COUNT) + 1);
        const width = 2 ** Math.ceil(Math.log2(length));
        this.allocate(count);
        this.device.queue.writeBuffer(this.buffers[0]!, 0, data);
        this.device.queue.writeBuffer(this.uniform, 0, new Uint32Array([count, width, 0, 0]));
        const encoder = this.device.createCommandEncoder();
        encoder.clearBuffer(this.buffers[4]!);
        const dispatch = (pipeline: number, group: number, work: number) => {
          const pass = encoder.beginComputePass();
          pass.setPipeline(this.pipelines[pipeline]!);
          pass.setBindGroup(0, this.groups[group]!);
          pass.dispatchWorkgroups(Math.ceil(work / 64));
          pass.end();
        };
        dispatch(0, 0, count * width * H); // Embedding writes buffer 2.
        let group = 1;
        for (let level = 0; 2 ** level < width; level++) {
          dispatch(1 + level, group, count * width * 2 * H);
          group = 1 - group;
        }
        dispatch(PROJECT, group, count * width * 32);
        dispatch(TOKEN_ROLES, group, count * width * R);
        dispatch(FAMILY_SCORES, group, count * F);
        const bytes = count * OUTPUT_COUNT * 4;
        encoder.copyBufferToBuffer(this.buffers[4]!, 0, this.readback!, 0, bytes);
        this.device.queue.submit([encoder.finish()]);
        await this.readback!.mapAsync(GPUMapMode.READ, 0, bytes);
        try {
          output.set(
            new Float32Array(this.readback!.getMappedRange(0, bytes)),
            start * OUTPUT_COUNT,
          );
        } finally {
          this.readback!.unmap();
        }
      }
      this.outputLocations = ['gpu-buffer'];
      return output;
    });
    this.pending = result.catch(() => undefined);
    return result;
  }
  dispose(): void {
    if (this.closed) return;
    this.closed = true;
    void this.pending.then(() => {
      this.buffers.forEach((b) => b.destroy());
      this.readback?.destroy();
      this.weights.destroy();
      this.uniform.destroy();
      this.device.destroy();
    });
  }
}

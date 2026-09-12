import weights from './weights.json' with { type: 'json' };
import { FEATURE_COUNT, HIDDEN_COUNT, FAMILIES, ROLES, tokenize } from '../features.js';
import type { Family } from '../types.js';

export const OUTPUT_COUNT = FAMILIES.length + FEATURE_COUNT * ROLES.length;
export const MODEL_INFO = Object.freeze({
  parameters: weights.parameters,
  format: 'packed-wgsl',
  architecture: weights.architecture,
  quantization: 'int6',
  features: FEATURE_COUNT,
  hidden: HIDDEN_COUNT,
  families: FAMILIES,
  roles: ROLES,
});
export const SEGMENTS = weights.segments;
export function decodeWeights(): Float32Array {
  const packed = atob(weights.data);
  const result = new Float32Array(weights.parameters);
  let bitBuffer = 0;
  let bitCount = 0;
  let byteIndex = 0;

  for (const segment of weights.segments) {
    for (let i = segment.offset; i < segment.offset + segment.length; i++) {
      while (bitCount < 6) {
        bitBuffer |= packed.charCodeAt(byteIndex++) << bitCount;
        bitCount += 8;
      }
      const unsignedValue = bitBuffer & 63;
      const signedValue = unsignedValue >= 32 ? unsignedValue - 64 : unsignedValue;
      result[i] = signedValue * segment.scale;
      bitBuffer >>>= 6;
      bitCount -= 6;
    }
  }
  return result;
}

function highestScore(scores: Float32Array): { index: number; confidence: number } {
  let index = 0;
  for (let i = 1; i < scores.length; i++) if (scores[i]! > scores[index]!) index = i;
  const sum = Array.from(scores).reduce((total, v) => total + Math.exp(v - scores[index]!), 0);
  return { index, confidence: 1 / sum };
}
export interface TokenPrediction {
  text: string;
  start: number;
  end: number;
  role: (typeof ROLES)[number];
  confidence: number;
}
/** Raw softmax scores are not calibrated correctness probabilities. */
export function prediction(
  logits: Float32Array,
  text: string,
): { family: Family; confidence: number; tokens: TokenPrediction[] } {
  const family = highestScore(logits.subarray(0, FAMILIES.length));
  return {
    family: FAMILIES[family.index]!,
    confidence: family.confidence,
    tokens: tokenize(text).map((token, i) => {
      const role = highestScore(
        logits.subarray(
          FAMILIES.length + i * ROLES.length,
          FAMILIES.length + (i + 1) * ROLES.length,
        ),
      );
      return {
        ...token,
        role: ROLES[role.index]!,
        confidence: role.confidence,
      };
    }),
  };
}

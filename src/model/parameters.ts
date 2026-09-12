import weights from './weights.json' with { type: 'json' };
import { FEATURE_COUNT, HIDDEN_COUNT, FAMILIES, ROLES, tokenize } from '../features.js';
import type { Family } from '../types.js';

export const OUTPUT_COUNT = FAMILIES.length + FEATURE_COUNT * ROLES.length;
export const MODEL_INFO = Object.freeze({ parameters: weights.parameters, format: 'packed-wgsl', architecture: weights.architecture, quantization: 'int6', features: FEATURE_COUNT, hidden: HIDDEN_COUNT, families: FAMILIES, roles: ROLES });
export const SEGMENTS = weights.segments;
export function decodeWeights(): Float32Array {
  const bytes = atob(weights.data);
  const result = new Float32Array(weights.parameters);
  let bits = 0, available = 0, cursor = 0;
  for (const segment of weights.segments) {
    for (let i = segment.offset; i < segment.offset + segment.length; i++) {
      while (available < 6) { bits |= bytes.charCodeAt(cursor++) << available; available += 8; }
      const q = bits & 63; bits >>>= 6; available -= 6;
      result[i] = (q >= 32 ? q - 64 : q) * segment.scale;
    }
  }
  return result;
}
function best(scores: Float32Array): {index:number; confidence:number} {
  let index = 0;
  for (let i=1;i<scores.length;i++) if(scores[i]! > scores[index]!) index=i;
  const sum = Array.from(scores).reduce((total,v)=>total+Math.exp(v-scores[index]!),0);
  return {index,confidence:1/sum};
}
export interface TokenPrediction {text:string;start:number;end:number;role:typeof ROLES[number];confidence:number}
/** Raw softmax scores are not calibrated correctness probabilities. */
export function prediction(logits: Float32Array, text = ''): {family:Family;confidence:number;tokens:TokenPrediction[]} {
  const family = best(logits.subarray(0,7));
  return {family:FAMILIES[family.index]!,confidence:family.confidence,tokens:tokenize(text).map((token,i)=>{
    const role = best(logits.subarray(7+i*ROLES.length,7+(i+1)*ROLES.length));
    return {...token,role:ROLES[role.index]!,confidence:role.confidence};
  })};
}

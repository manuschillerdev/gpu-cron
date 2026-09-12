// Development-only numerical reference. Never imported by the browser library.
import {decodeWeights,SEGMENTS,OUTPUT_COUNT} from '../dist/model/parameters.js';
import {FEATURE_COUNT,HIDDEN_COUNT,ROLES,featureRows} from '../dist/features.js';
const w=decodeWeights(),H=HIDDEN_COUNT,R=ROLES.length;
const at=name=>SEGMENTS.find(s=>s.name===name).offset;
const E=at('embedding.weight'),A=at('affine.weight'),AB=at('affine.bias'),W=at('hidden.weight'),WB=at('hidden.bias'),O=at('output.weight'),OB=at('output.bias');
function embedding(id,k){return id ? Math.fround(featureRows(id).reduce((s,r)=>s+w[E+r*H+k],0)/Math.sqrt(3)) : 0;}
export function cpuLogits(input){
  if(input.length%FEATURE_COUNT)throw new Error('Invalid model input shape.');
  const output=new Float32Array(input.length/FEATURE_COUNT*OUTPUT_COUNT);
  for(let row=0;row<input.length/FEATURE_COUNT;row++){
    const ids=input.subarray(row*FEATURE_COUNT,(row+1)*FEATURE_COUNT);
    let n=ids.length;while(n && !ids[n-1])n--;
    if(!n)continue;
    const a=new Float32Array(n*H),b=new Float32Array(n*H),f=new Float32Array(n*H),r=new Float32Array(n*H);
    for(let t=0;t<n;t++)for(let c=0;c<H;c++){
      let ga=w[AB+c],gb=w[AB+H+c];
      for(let k=0;k<H;k++){const e=embedding(ids[t],k);ga+=e*w[A+c*H+k];gb+=e*w[A+(c+H)*H+k];}
      a[t*H+c]=1/(1+Math.exp(-ga));b[t*H+c]=Math.tanh(gb);
    }
    for(let c=0;c<H;c++){
      let left=0,right=0;
      for(let t=0;t<n;t++){left=a[t*H+c]*left+b[t*H+c];f[t*H+c]=left;const j=n-1-t;right=a[j*H+c]*right+b[j*H+c];r[j*H+c]=right;}
    }
    const hidden=new Float32Array(32),family=new Float64Array(7);
    for(let t=0;t<n;t++){
      for(let c=0;c<32;c++){
        let sum=w[WB+c];for(let k=0;k<H;k++){sum+=embedding(ids[t],k)*w[W+c*3*H+k];sum+=f[t*H+k]*w[W+c*3*H+H+k];sum+=r[t*H+k]*w[W+c*3*H+2*H+k];}hidden[c]=Math.max(0,sum);
      }
      for(let c=0;c<7+R;c++){
        let sum=w[OB+c];for(let k=0;k<32;k++)sum+=hidden[k]*w[O+c*32+k];
        if(c<7)family[c]+=sum;else output[row*OUTPUT_COUNT+7+t*R+c-7]=sum;
      }
    }
    for(let c=0;c<7;c++)output[row*OUTPUT_COUNT+c]=family[c]/n;
  }
  return output;
}

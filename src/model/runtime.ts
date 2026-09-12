import {decodeWeights, SEGMENTS, OUTPUT_COUNT} from './parameters.js';
import {FEATURE_COUNT, HIDDEN_COUNT, ROLES} from '../features.js';

const H = HIDDEN_COUNT;
const R = ROLES.length;
const offset = (name:string) => {
  const segment = SEGMENTS.find(s=>s.name===name);
  if (!segment) throw new Error(`Missing model tensor ${name}`);
  return segment.offset;
};
/** Constants and operators are specialized for this scan network at initialization. */
function shaders(): string[] {
  const E=offset('embedding.weight'), A=offset('affine.weight'), AB=offset('affine.bias');
  const W=offset('hidden.weight'), WB=offset('hidden.bias'), O=offset('output.weight'), OB=offset('output.bias');
  const common=`@group(0) @binding(0) var<storage,read> ids:array<f32>;
@group(0) @binding(1) var<storage,read> w:array<f32>;
@group(0) @binding(2) var<storage,read> src:array<f32>;
@group(0) @binding(3) var<storage,read_write> dst:array<f32>;
@group(0) @binding(4) var<storage,read_write> hidden:array<f32>;
@group(0) @binding(5) var<storage,read_write> out:array<f32>;
@group(0) @binding(6) var<uniform> p:vec4<u32>;
fn embedding(id:u32,k:u32)->f32 {
 if(id==0u){return 0.0;}let bits=id-1u;
 return (w[${E}u+(bits&1023u)*${H}u+k]+w[${E}u+(1024u+((bits>>10u)&255u))*${H}u+k]+w[${E}u+(1280u+((bits>>18u)&31u))*${H}u+k])*0.5773502691896258;
}
fn token(i:u32)->u32 { return u32(ids[(i/p.y)*${FEATURE_COUNT}u+i%p.y]); }
`;
  const kernel=(body:string)=>common+`@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){${body}}`;
  const embed=kernel(`let i=g.x/${H}u; let c=g.x%${H}u; if(i>=p.x*p.y){return;}
let id=token(i); var a=1.0; var b=0.0;
if(id!=0u){ var ga=w[${AB}u+c]; var gb=w[${AB+H}u+c];
for(var k=0u;k<${H}u;k++){let e=embedding(id,k); ga+=e*w[${A}u+c*${H}u+k]; gb+=e*w[${A+H*H}u+c*${H}u+k];}
a=1.0/(1.0+exp(-ga)); b=tanh(gb); }
let base=i*${4*H}u+c*2u;dst[base]=a;dst[base+1u]=b;dst[base+${2*H}u]=a;dst[base+${2*H+1}u]=b;`);
  const scans=Array.from({length:9},(_,level)=>kernel(`let i=g.x/${2*H}u;let c=g.x%${2*H}u;if(i>=p.x*p.y){return;}
let pos=i%p.y;let base=i*${4*H}u+c*2u;var a=src[base];var b=src[base+1u];
let forward=c<${H}u;let valid=select(pos+${2**level}u<p.y,pos>=${2**level}u,forward);
if(valid){let other=select(i+${2**level}u,i-${2**level}u,forward)*${4*H}u+c*2u;b+=a*src[other+1u];a*=src[other];}
dst[base]=a;dst[base+1u]=b;`));
  const project=kernel(`let i=g.x/32u;let c=g.x%32u;if(i>=p.x*p.y){return;}
let id=token(i);var sum=w[${WB}u+c];
for(var k=0u;k<${H}u;k++){
sum+=select(embedding(id,k),0.0,id==0u)*w[${W}u+c*${3*H}u+k];
sum+=src[i*${4*H}u+k*2u+1u]*w[${W+H}u+c*${3*H}u+k];
sum+=src[i*${4*H}u+${2*H}u+k*2u+1u]*w[${W+2*H}u+c*${3*H}u+k];}
hidden[i*32u+c]=max(0.0,sum);`);
  const roles=kernel(`let i=g.x/${R}u;let c=g.x%${R}u;if(i>=p.x*p.y){return;}
var sum=w[${OB+7}u+c];for(var k=0u;k<32u;k++){sum+=hidden[i*32u+k]*w[${O+7*32}u+c*32u+k];}
out[(i/p.y)*${OUTPUT_COUNT}u+7u+(i%p.y)*${R}u+c]=select(sum,0.0,token(i)==0u);`);
  const family=kernel(`let row=g.x/7u;let c=g.x%7u;if(row>=p.x){return;}
var total=0.0;var count=0.0;for(var t=0u;t<p.y;t++){let i=row*p.y+t;if(token(i)==0u){continue;}
var sum=w[${OB}u+c];for(var k=0u;k<32u;k++){sum+=hidden[i*32u+k]*w[${O}u+c*32u+k];}total+=sum;count+=1.0;}
out[row*${OUTPUT_COUNT}u+c]=total/max(count,1.0);`);
  return [embed,...scans,project,roles,family];
}

export class CronModel {
  outputLocations:string[]=[];
  readonly executionProvider='webgpu' as const;
  private closed=false;
  private pending:Promise<unknown>=Promise.resolve();
  private lost?:string;
  private buffers:GPUBuffer[]=[];
  private groups:GPUBindGroup[]=[];
  private readback?:GPUBuffer;
  private capacity=0;
  private constructor(private device:GPUDevice,private weights:GPUBuffer,private uniform:GPUBuffer,private layout:GPUBindGroupLayout,private pipelines:GPUComputePipeline[]){
    void device.lost.then(info=>{this.lost=info.message||'WebGPU device lost';});
  }
  static async create():Promise<CronModel>{
    const adapter=await globalThis.navigator?.gpu?.requestAdapter();
    if(!adapter)throw new Error('WebGPU is required. Use a browser and device with WebGPU support.');
    const device=await adapter.requestDevice();
    try{
      const layout=device.createBindGroupLayout({entries:Array.from({length:7},(_,binding)=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===6?'uniform':binding<3?'read-only-storage':'storage'}}))});
      const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
      const pipelines=await Promise.all(shaders().map(code=>device.createComputePipelineAsync({layout:pipelineLayout,compute:{module:device.createShaderModule({code}),entryPoint:'main'}})));
      const values=decodeWeights();
      const weights=device.createBuffer({size:values.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
      device.queue.writeBuffer(weights,0,values as Float32Array<ArrayBuffer>);
      const uniform=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
      return new CronModel(device,weights,uniform,layout,pipelines);
    }catch(error){device.destroy();throw error;}
  }
  private allocate(rows:number):void{
    if(rows<=this.capacity)return;
    const capacity=2**Math.ceil(Math.log2(rows));
    const widths=[FEATURE_COUNT,FEATURE_COUNT*4*H,FEATURE_COUNT*4*H,FEATURE_COUNT*32,OUTPUT_COUNT];
    if(widths.some(w=>capacity*w*4>this.device.limits.maxStorageBufferBindingSize))throw new RangeError('Batch exceeds WebGPU limits');
    this.buffers.forEach(b=>b.destroy());this.readback?.destroy();
    this.buffers=widths.map(w=>this.device.createBuffer({size:capacity*w*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}));
    this.readback=this.device.createBuffer({size:capacity*OUTPUT_COUNT*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
    this.groups=[0,1].map(swap=>this.device.createBindGroup({layout:this.layout,entries:[this.buffers[0]!,this.weights,this.buffers[1+swap]!,this.buffers[2-swap]!,this.buffers[3]!,this.buffers[4]!,this.uniform].map((buffer,binding)=>({binding,resource:{buffer}}))}));
    this.capacity=capacity;
  }
  logits(input:Float32Array):Promise<Float32Array>{
    if(this.closed)return Promise.reject(new Error('Model is disposed.'));
    if(input.length%FEATURE_COUNT)return Promise.reject(new Error('Invalid model input shape.'));
    const snapshot=new Float32Array(input);
    const result=this.pending.then(async()=>{
      if(this.lost)throw new Error(this.lost);
      const rows=snapshot.length/FEATURE_COUNT;
      if(!rows)return new Float32Array();
      const output=new Float32Array(rows*OUTPUT_COUNT);
      // Bound storage use for the public 4096-expression batch limit.
      for(let start=0;start<rows;start+=128){
        const count=Math.min(128,rows-start),data=snapshot.subarray(start*FEATURE_COUNT,(start+count)*FEATURE_COUNT);
        let length=1;for(let i=0;i<data.length;i++)if(data[i]!==0)length=Math.max(length,i%FEATURE_COUNT+1);
        const width=2**Math.ceil(Math.log2(length));
        this.allocate(count);
        this.device.queue.writeBuffer(this.buffers[0]!,0,data);
        this.device.queue.writeBuffer(this.uniform,0,new Uint32Array([count,width,0,0]));
        const encoder=this.device.createCommandEncoder();encoder.clearBuffer(this.buffers[4]!);
        const dispatch=(pipeline:number,group:number,work:number)=>{const pass=encoder.beginComputePass();pass.setPipeline(this.pipelines[pipeline]!);pass.setBindGroup(0,this.groups[group]!);pass.dispatchWorkgroups(Math.ceil(work/64));pass.end();};
        dispatch(0,0,count*width*H); // Embedding writes buffer 2.
        let group=1;
        for(let level=0;2**level<width;level++){dispatch(1+level,group,count*width*2*H);group=1-group;}
        dispatch(10,group,count*width*32);dispatch(11,group,count*width*R);dispatch(12,group,count*7);
        const bytes=count*OUTPUT_COUNT*4;encoder.copyBufferToBuffer(this.buffers[4]!,0,this.readback!,0,bytes);
        this.device.queue.submit([encoder.finish()]);
        await this.readback!.mapAsync(GPUMapMode.READ,0,bytes);
        try{output.set(new Float32Array(this.readback!.getMappedRange(0,bytes).slice(0)),start*OUTPUT_COUNT);}finally{this.readback!.unmap();}
      }
      this.outputLocations=['gpu-buffer'];return output;
    });
    this.pending=result.catch(()=>undefined);return result;
  }
  dispose():void{
    if(this.closed)return;this.closed=true;
    void this.pending.then(()=>{this.buffers.forEach(b=>b.destroy());this.readback?.destroy();this.weights.destroy();this.uniform.destroy();this.device.destroy();});
  }
}

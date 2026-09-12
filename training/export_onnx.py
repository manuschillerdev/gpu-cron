"""Development-only portable graph of the exact shipped six-bit scan network."""
import base64
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
from data import MAX_TOKENS, WIDTH, ROLES

ROOT=Path(__file__).resolve().parents[1]

def main():
    source=ROOT/'src/model/weights.json'
    payload=json.loads(source.read_text())
    raw=base64.b64decode(payload['data'])
    values=[];bits=available=cursor=0
    for _ in range(payload['parameters']):
        while available<6:
            bits|=raw[cursor]<<available;available+=8;cursor+=1
        q=bits&63;bits>>=6;available-=6
        values.append(q-64 if q>=32 else q)
    init=[];nodes=[];serial=0
    def const(value,dtype=np.int64,name=None):
        nonlocal serial
        serial+=1;name=name or f'constant_{serial}'
        init.append(nh.from_array(np.asarray(value,dtype=dtype),name));return name
    def op(kind,*inputs,**attrs):
        nonlocal serial
        serial+=1;name=f'{kind}_{serial}'
        nodes.append(h.make_node(kind,list(inputs),[name],**attrs));return name
    for s in payload['segments']:
        arr=(np.asarray(values[s['offset']:s['offset']+s['length']],dtype=np.float64)*s['scale']).astype(np.float32).reshape(s['shape'])
        init.append(nh.from_array(arr,s['name']))
    def linear(name,x):
        return op('Add',op('MatMul',x,op('Transpose',name+'.weight',perm=[1,0])),name+'.bias')
    def slice_(x,start,end,axis):
        return op('Slice',x,const([start]),const([end]),const([axis]))
    def scan(a,b):
        offset=1
        while offset<MAX_TOKENS:
            tail=op('Add',slice_(b,offset,MAX_TOKENS,1),op('Mul',slice_(a,offset,MAX_TOKENS,1),slice_(b,0,MAX_TOKENS-offset,1)))
            b=op('Concat',slice_(b,0,offset,1),tail,axis=1)
            a=op('Concat',slice_(a,0,offset,1),op('Mul',slice_(a,offset,MAX_TOKENS,1),slice_(a,0,MAX_TOKENS-offset,1)),axis=1)
            offset*=2
        return b
    ids=op('Cast','features',to=T.INT64)
    mask=op('Not',op('Equal',ids,const(0)))
    mask3=op('Unsqueeze',mask,const([2]))
    maskf=op('Cast',mask3,to=T.FLOAT)
    bits=op('Max',op('Sub',ids,const(1)),const(0))
    rows=[op('Mod',bits,const(1024)),op('Add',const(1024),op('Mod',op('Div',bits,const(1024)),const(256))),op('Add',const(1280),op('Mod',op('Div',bits,const(262144)),const(32)))]
    embeds=[op('Gather','embedding.weight',r,axis=0) for r in rows]
    e=op('Mul',op('Mul',op('Add',op('Add',embeds[0],embeds[1]),embeds[2]),const(1/3**.5,np.float32)),maskf)
    ab=linear('affine',e)
    a=op('Where',mask3,op('Sigmoid',slice_(ab,0,WIDTH,2)),const(1,np.float32))
    b=op('Where',mask3,op('Tanh',slice_(ab,WIDTH,2*WIDTH,2)),const(0,np.float32))
    reverse=const(np.arange(MAX_TOKENS-1,-1,-1))
    forward=scan(a,b)
    backward=op('Gather',scan(op('Gather',a,reverse,axis=1),op('Gather',b,reverse,axis=1)),reverse,axis=1)
    hidden=op('Relu',linear('hidden',op('Concat',e,forward,backward,axis=2)))
    scores=op('Mul',linear('output',hidden),maskf)
    family=op('Div',op('ReduceSum',slice_(scores,0,7,2),const([1]),keepdims=0),op('Max',op('ReduceSum',maskf,const([1]),keepdims=0),const(1,np.float32)))
    tags=op('Reshape',slice_(scores,7,7+len(ROLES),2),const([-1,MAX_TOKENS*len(ROLES)]))
    out=op('Concat',family,tags,axis=1)
    nodes.append(h.make_node('Identity',[out],['logits']))
    graph=h.make_graph(nodes,'gpu-cron-six-bit-bidirectional-scan',[h.make_tensor_value_info('features',T.FLOAT,['batch',MAX_TOKENS])],[h.make_tensor_value_info('logits',T.FLOAT,['batch',7+MAX_TOKENS*len(ROLES)])],initializer=init)
    model=h.make_model(graph,producer_name='gpu-cron',opset_imports=[h.make_opsetid('',17)],ir_version=10)
    onnx.checker.check_model(model,full_check=True)
    path=ROOT/'models/cron.onnx';onnx.save_model(model,path)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report={'sourceWeights':'src/model/weights.json','sourceWeightsSha256':sha(source),'modelSha256':sha(path),'exportSourceSha256':sha(Path(__file__)),'modelBytes':path.stat().st_size,'parameters':payload['parameters'],'onnxCheckerPassed':True,'onnxVersion':onnx.__version__,'architecture':payload['architecture'],'input':{'name':'features','shape':['batch',MAX_TOKENS],'dtype':'float32 packed feature bits'},'output':{'name':'logits','shape':['batch',7+MAX_TOKENS*len(ROLES)]},'weights':'Exact decoded shipped int6 coefficients. Development verification only.'}
    (ROOT/'models/cron-export.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

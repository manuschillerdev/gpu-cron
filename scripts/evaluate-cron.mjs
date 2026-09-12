/** Dataset evaluation, separate from regression tests. Inference is actual WebGPU. */
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {createServer} from 'vite';
import {chromium} from 'playwright';

const holdout=process.argv.includes('--holdout');
const names=holdout?['patternHoldout','authoredHoldout']:['development'];
const server=await createServer({server:{host:'127.0.0.1',port:4190,strictPort:false},logLevel:'error'});
let browser;
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
try{
  const training=JSON.parse(await readFile('training/report.json','utf8'));
  if(sha(await readFile('training/data/manifest.json'))!==training.manifestSha256
    ||sha(await readFile('src/model/vocabulary.json'))!==training.vocabularySha256
    ||sha(await readFile('src/model/weights.json'))!==training.weightsSha256)throw new Error('Dataset/model identity differs from the recorded training run. Retrain or restore the matching artifacts before evaluation.');
  await server.listen();
  browser=await chromium.launch({channel:'chromium',headless:true,args:['--enable-unsafe-webgpu']});
  const page=await browser.newPage();await page.goto(server.resolvedUrls.local[0]);
  await page.waitForFunction(()=>document.querySelector('#status').textContent.startsWith('WEBGPU'));
  const reports={};
  for(const name of names){
    const file=`training/data/${name==='authoredHoldout'?'authored':name}.jsonl`;
    const bytes=await readFile(file);const records=bytes.toString().trim().split('\n').filter(Boolean).map(JSON.parse);
    const outcomes=await page.evaluate(async records=>{
      const {defineParser}=await import('/src/index.ts');
      const {compile}=await import('/src/compile.ts');
      const {tokenize}=await import('/src/features.ts');
      const parser=defineParser();const results=[];
      const canonical=s=>s?Object.fromEntries(['minutes','hours','daysOfMonth','weekdays','months','weekInterval'].map(k=>[k,Array.isArray(s[k])?[...s[k]].sort((a,b)=>a-b):s[k]])):null;
      const equal=(a,b)=>JSON.stringify(canonical(a))===JSON.stringify(canonical(b));
      try{
        for(let start=0;start<records.length;start+=128){
          const batch=records.slice(start,start+128);
          const predictions=await parser.parseMany(batch.map(r=>r.text),{timeZone:'UTC',reference:'2026-09-14T00:00:00Z',count:0});
          predictions.forEach((pred,i)=>{
            const gold=batch[i];
            const actualTokens=tokenize(gold.text);
            if(actualTokens.length!==gold.tokens.length || actualTokens.some((t,j)=>t.start!==gold.tokens[j].start || t.end!==gold.tokens[j].end))throw new Error(`Annotation/token alignment mismatch: ${gold.id}`);
            const goldTokens=gold.tokens.map(t=>({...t,text:gold.text.slice(t.start,t.end),confidence:1}));
            const oracle=compile(gold.text,gold.family,'UTC','2026-09-14',goldTokens);
            const tokenCorrect=gold.tokens.filter((t,j)=>pred.model.tokens[j]?.role===t.role).length;
            const familyCorrect=pred.model.family===gold.family;
            const spansCorrect=tokenCorrect===gold.tokens.length && pred.model.tokens.length===gold.tokens.length;
            const exact=equal(pred.schedule,gold.target.schedule);
            const oracleExact=equal(oracle.schedule,gold.target.schedule);
            const failure=exact?null:gold.target.status!=='supported'?'accepted-ambiguous-or-unsupported':oracleExact?'model':familyCorrect&&spansCorrect?'compiler':'model-and-compiler';
            results.push({id:gold.id,groupId:gold.groupId,category:gold.category,text:gold.text,target:gold.target,
              network:{familyCorrect,spansCorrect,tokenCorrect,tokenTotal:gold.tokens.length,prediction:pred.model},
              compiler:{exact,schedule:pred.schedule,diagnostics:pred.diagnostics},
              oracleCompiler:{exact:oracleExact,schedule:oracle.schedule,diagnostics:oracle.diagnostics},failure});
          });
        }
      }finally{parser.dispose();}
      return results;
    },records);
    const summarize=rows=>{
      const count=rows.length;const ratio=n=>count?n/count:null;
      const supported=rows.filter(r=>r.target.status==='supported');
      const ambiguous=rows.filter(r=>r.target.status==='ambiguous');
      const unsupported=rows.filter(r=>r.target.status==='unsupported');
      const tokenTotal=rows.reduce((sum,r)=>sum+r.network.tokenTotal,0);
      return {examples:count,meaningGroups:new Set(rows.map(r=>r.groupId)).size,
        network:{familyAccuracy:ratio(rows.filter(r=>r.network.familyCorrect).length),exactSpanAccuracy:ratio(rows.filter(r=>r.network.spansCorrect).length),tokenAccuracy:tokenTotal?rows.reduce((s,r)=>s+r.network.tokenCorrect,0)/tokenTotal:null},
        compiler:{exactStructuredScheduleAccuracy:supported.length?supported.filter(r=>r.compiler.exact).length/supported.length:null,supportedExamples:supported.length,correctOutcomeRate:ratio(rows.filter(r=>r.compiler.exact).length),unsupportedExamples:unsupported.length,unsupportedRejectionRate:unsupported.length?unsupported.filter(r=>!r.compiler.schedule).length/unsupported.length:null,ambiguousExamples:ambiguous.length,ambiguousRejectionRate:ambiguous.length?ambiguous.filter(r=>!r.compiler.schedule).length/ambiguous.length:null},
        oracleCompiler:{exactStructuredScheduleAccuracy:supported.length?supported.filter(r=>r.oracleCompiler.exact).length/supported.length:null},
        errors:Object.fromEntries(['model','compiler','model-and-compiler','accepted-ambiguous-or-unsupported'].map(k=>[k,rows.filter(r=>r.failure===k).length]))};
    };
    const summary=summarize(outcomes);
    const report={split:name,sourceSha256:sha(bytes),compilerSha256:sha(await readFile('src/compile.ts')),featuresSha256:sha(await readFile('src/features.ts')),runtimeSha256:sha(await readFile('src/model/runtime.ts')),weightsSha256:sha(await readFile('src/model/weights.json')),vocabularySha256:training.vocabularySha256,manifestSha256:sha(await readFile('training/data/manifest.json')),evaluationUse:'Development: prior results informed this iteration; these are not untouched holdouts.',annotationVersion:2,executionProvider:'webgpu',context:{timeZone:'UTC',reference:'2026-09-14T00:00:00Z'},...summary,
      byCategory:Object.fromEntries([...new Set(outcomes.map(r=>r.category))].sort().map(c=>[c,summarize(outcomes.filter(r=>r.category===c))])),
      interpretation:'Network labels, actual compiler output, and compiler output given annotated labels are scored separately. Canonical schedule comparison excludes redundant family labels and fixes timezone/alternate-week context. Authored requests are assistant-written, not real-user traffic. Evaluation requests are not fitted by the trainer; prior evaluation findings informed generator and compiler development. Token labels changed from coarse clauses to semantic roles, so token accuracy is not directly comparable with the old model.'};
    await mkdir('training/evaluation',{recursive:true});
    await writeFile(`training/evaluation/${name}.json`,JSON.stringify(report,null,2)+'\n');
    await mkdir('test-artifacts',{recursive:true});
    await writeFile(`test-artifacts/cron-${name}-predictions.json`,JSON.stringify(outcomes,null,2)+'\n');
    reports[name]=summary;
  }
  console.log(JSON.stringify(reports,null,2));
}finally{await browser?.close();await server.close();}

export async function checkRuntimeLifecycle(page) {
  return page.evaluate(async () => {
    const {CronModel} = await import('/src/model/runtime.ts');
    const {cpuLogits} = await import('/scripts/cron-reference.mjs');
    const {features} = await import('/src/features.ts');
    const fixtures = await (await fetch('/test/model-fixtures.json')).json();
    const inputs = fixtures.slice(0,4).map(f=>features(f.text));
    const batch = count => new Float32Array(inputs.slice(0,count).flatMap(v=>Array.from(v)));
    const createBuffer = GPUDevice.prototype.createBuffer;
    let allocations = 0;
    GPUDevice.prototype.createBuffer = function(...args) { allocations++; return createBuffer.apply(this,args); };
    const runtime = await CronModel.create();
    let result;
    try {
      const input = batch(2);
      const expected = cpuLogits(input);
      await runtime.logits(input);
      const warmed = allocations;
      const pending = runtime.logits(input);
      input.fill(100); // queued calls must retain their original input
      const preserved = await pending;
      await runtime.logits(inputs[0]);
      const reused = allocations === warmed;
      await runtime.logits(batch(4));
      const grew = allocations > warmed;
      const first = runtime.logits(inputs[0]);
      const second = runtime.logits(inputs[1]);
      const disposal = runtime.dispose();
      const values = await Promise.all([first,second]);
      await disposal; await runtime.dispose();
      let rejected = false;
      try { await runtime.logits(inputs[0]); } catch { rejected = true; }
      result = {reused,grew,rejected,snapshot:preserved.every((v,i)=>Math.abs(v-expected[i])<1e-4),queued:values.every((v,i)=>v.every((score,j)=>Math.abs(score-expected[i*v.length+j])<1e-4))};
    } finally { GPUDevice.prototype.createBuffer = createBuffer; await runtime.dispose(); }
    return result;
  });
}

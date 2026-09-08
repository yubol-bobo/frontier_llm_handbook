import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {tokenStep,batchComparison,validateProgress,searchDocuments} from '../learning.mjs';

const near=(a,b,tolerance=1e-9)=>assert.ok(Math.abs(a-b)<tolerance,`${a} != ${b}`);
const vectorNear=(a,b,tolerance)=>a.forEach((v,i)=>near(v,b[i],tolerance));

test('browser teaching math agrees with the independently executed Python experiment',async()=>{
  const record=JSON.parse(await fs.readFile(new URL('../../experiments/002-token-weighted-loss/results.json',import.meta.url),'utf8'));
  const expected=record.single_position;
  const r=tokenStep(expected.logits,expected.target,expected.learning_rate);
  vectorNear(r.probabilities,expected.probabilities);
  near(r.loss,expected.loss);
  vectorNear(r.gradient,expected.gradient);
  near(tokenStep(r.next,expected.target,0).loss,expected.loss_after_step);
  near(r.nextLoss,expected.loss_after_step);
  const batch=batchComparison(expected.logits);
  near(batch.global.loss,record.batch.loss);
  vectorNear(batch.global.gradient,record.batch.gradient);
});

test('gradient matches finite differences for every target and several logit shapes',()=>{
  for(const z of [[2,1,0],[0,0,0],[-3,2,1],[10,-10,4]])for(const target of [0,1,2]){
    const r=tokenStep(z,target,0),h=1e-5;
    for(let j=0;j<3;j++){
      const plus=[...z],minus=[...z];plus[j]+=h;minus[j]-=h;
      near((tokenStep(plus,target,0).loss-tokenStep(minus,target,0).loss)/(2*h),r.gradient[j],1e-8);
    }
  }
});

test('stable logsumexp handles translation, underflow and invalid inputs',()=>{
  const base=tokenStep([2,1,0],1,0),shift=tokenStep([1002,1001,1000],1,0);
  near(base.loss,shift.loss);vectorNear(base.probabilities,shift.probabilities);
  const extreme=tokenStep([10000,-10000,0],1,0);near(extreme.loss,20000);assert.equal(extreme.probabilities[1],0);
  const boundary=tokenStep([10000,10000,10000],1,10);near(boundary.loss,Math.log(3));assert.ok(Number.isFinite(boundary.nextLoss));assert.ok(boundary.next[1]>10000);
  vectorNear(tokenStep([2,1,0],1,0).next,[2,1,0]);
  for(const args of [[[NaN,1,0],1,.1],[[Infinity,1,0],1,.1],[[2,1,0],3,.1],[[2,1,0],1,-1],[[2,1,0],1,NaN]])assert.throws(()=>tokenStep(...args));
});

test('all mask patterns retain the global objective across all partitions',()=>{
  for(let bits=1;bits<64;bits++){
    const masks=Array.from({length:6},(_,i)=>(bits>>i)&1);
    const base=batchComparison([2,1,0],masks,'uneven').global;
    for(const partition of ['uneven','balanced','empty']){
      const b=batchComparison([2,1,0],masks,partition);
      near(b.global.loss,base.loss);near(b.weighted.loss,base.loss);vectorNear(b.weighted.gradient,base.gradient);
    }
  }
});

test('uneven normalization counterexample and empty-target semantics',()=>{
  const b=batchComparison([2,1,0]);
  near(b.global.loss,1.9076059644443804);near(b.equalGroups.loss,1.4076059644443804);
  vectorNear(b.global.gradient,[.4152409557748218,.24472847105479764,-.6599694268296195]);
  vectorNear(b.equalGroups.gradient,[.1652409557748218,.24472847105479764,-.4099694268296195]);
  const emptyGroup=batchComparison([2,1,0],[1,1,1,1,0,0],'empty');assert.equal(emptyGroup.groups[1].loss,null);
  const none=batchComparison([2,1,0],[0,0,0,0,0,0]);assert.equal(none.global,null);assert.equal(none.count,0);
});

test('portable progress rejects malformed records and unsupported completion claims',()=>{
  const good={schemaVersion:1,lastModule:'M02',modules:{M02:{status:'learning',note:'梯度验证',evidence:'',updated:'2026-09-08T00:00:00Z'}}};
  assert.deepEqual(validateProgress(JSON.parse(JSON.stringify(good))),good);
  assert.throws(()=>validateProgress({...good,schemaVersion:99}));
  assert.throws(()=>validateProgress({...good,modules:[]}));
  assert.throws(()=>validateProgress({...good,modules:{M99:good.modules.M02}}));
  assert.throws(()=>validateProgress({...good,modules:{M02:{...good.modules.M02,status:'reviewed'}}}));
  const valid=validateProgress({...good,modules:{M02:{...good.modules.M02,status:'reviewed',evidence:'有限差分脚本与记录'}}});assert.equal(valid.modules.M02.status,'reviewed');
  assert.throws(()=>validateProgress({...good,modules:{M02:{...good.modules.M02,note:'x'.repeat(20001)}}}));
});

test('search supports Chinese, multiple terms, ranking and empty queries',()=>{
  const docs={a:{path:'a.md',title:'Checkpoint 恢复',markdown:'恢复 checkpoint 与状态'},b:{path:'b.md',title:'系统',markdown:'checkpoint 与恢复'},c:{path:'c.md',title:'训练',markdown:'loss mask'}};
  assert.equal(searchDocuments(docs,'Checkpoint 恢复')[0].path,'a.md');
  assert.equal(searchDocuments(docs,'loss mask').length,1);assert.equal(searchDocuments(docs,'不存在').length,0);assert.equal(searchDocuments(docs,' ').length,0);
});

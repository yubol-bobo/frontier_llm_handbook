// CPU exercise against the pinned upstream economics function, with no model calls.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';

const directory=path.dirname(fileURLToPath(import.meta.url));
const source=path.resolve(directory,'../../sources/sol-pi');
const expected='d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0';
assert.equal(execFileSync('git',['rev-parse','HEAD'],{cwd:source,encoding:'utf8'}).trim(),expected,'Restore the recorded source snapshot before this exercise.');
assert.equal(execFileSync('git',['status','--porcelain','--untracked-files=no'],{cwd:source,encoding:'utf8'}).trim(),'','Tracked source must be unchanged.');
const {decideCompaction,DEFAULT_COMPACTION_ECONOMICS}=await import(pathToFileURL(path.join(source,'src/sol-pi/extensions/online-context-compact/economics.ts')));

const base={writeTokens:80000,archiveTokens:60000,memoTokens:1000,contextTokens:80000,
  completedBoundaryRequestCounts:[4,6,5],remainingBoundaries:4,averageContextTokenIncrement:2000,
  contextWindowTokens:200000,priorCompactionCount:0,carriedDebtTokens:0,cacheDebtRepaymentTokens:0,
  cacheWriteReadRatio:12.5,economics:DEFAULT_COMPACTION_ECONOMICS};
const scenarios=[
  ['first',{},true,'economic'],
  ['later_margin',{priorCompactionCount:1},false,'deferred_subsequent_margin'],
  ['later_longer_horizon',{priorCompactionCount:1,remainingBoundaries:6},true,'economic'],
  ['unpaid_debt',{priorCompactionCount:1,remainingBoundaries:6,carriedDebtTokens:2000000},false,'deferred_carried_debt'],
  ['window_pressure',{writeTokens:190000,archiveTokens:150000,contextTokens:190000,cacheWriteReadRatio:100},true,'window_protection'],
  ['no_saving',{archiveTokens:1000,memoTokens:1000},false,'non_positive_saving'],
  ['no_history',{completedBoundaryRequestCounts:null},false,'horizon_unavailable'],
  ['no_price_ratio',{cacheWriteReadRatio:null},false,'cache_ratio_unavailable'],
];
const results=scenarios.map(([name,changes,compact,reason])=>{
  const input={...base,...changes};
  const actual=decideCompaction(input);
  assert.equal(actual.compact,compact,name);
  assert.equal(actual.reason,reason,name);
  return {name,input,decision:actual};
});

// Independent arithmetic on this fixture, rather than computing expected values with upstream helpers.
const first=results[0].decision;
assert.equal(first.requestsPerBoundaryMean,5);
assert.equal(first.expectedRemainingRequests,21);
assert.equal(first.windowRequestUpperBound,60);
assert.equal(first.effectiveHorizonRequests,42);
assert.ok(Math.abs(first.breakevenRequests-920000/59000)<1e-10);
assert.ok(Math.abs(results[3].decision.combinedBreakevenRequests-2920000/59000)<1e-10);

await fs.writeFile(path.join(directory,'results.json'),JSON.stringify({
  recordedDate:new Date().toISOString().slice(0,10),node:process.version,platform:process.platform,sourceCommit:expected,
  scope:'8 deterministic scenarios invoking the real upstream economics function; no native compaction, model, API pricing measurement, or benchmark',
  casesPassed:results.length,results,
},null,2)+'\n');
console.log('PASS: 8 upstream compaction-economics scenarios and independent arithmetic. Results: experiments/004-sol-pi-contracts/results.json');

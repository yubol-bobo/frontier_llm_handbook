import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {renderLearningView,relationEdges,curveBetween,lossHistoryPoints,icon} from '../design.mjs';
import {tokenResultsShell,batchResultsHtml} from '../lab-view.mjs';
import {batchComparison} from '../learning.mjs';

const content=JSON.parse(await fs.readFile(new URL('../../_site/content.json',import.meta.url),'utf8'));
const props={data:content,progress:{lastModule:null},entry:()=>({status:'new'}),readLink:file=>'#/read/'+file,moduleLink:id=>'#/module/'+id,statusTag:id=>'<span>'+id+'</span>',stageFilter:'all'};

test('new course cards preserve all modules, seminars, filters and the true continuation',()=>{
  const html=renderLearningView(props);
  assert.equal([...html.matchAll(/class="module-card"/g)].length,16);
  assert.equal([...html.matchAll(/class="seminar"/g)].length,6);
  assert.equal([...html.matchAll(/class="stage-choice /g)].length,3);
  assert.ok(html.includes('href="#/module/M00"'));
  const filtered=renderLearningView({...props,stageFilter:'systems'});
  assert.equal([...filtered.matchAll(/class="module-card"/g)].length,5);
  assert.ok(!filtered.includes('<span class="module-code">M01</span>'));
  const continued=renderLearningView({...props,progress:{lastModule:'M08'}});
  assert.ok(continued.includes('<span class="continue-module">M08'));
  assert.ok(continued.includes('继续探索'));
});

test('topic membership does not become an invented prerequisite chain',()=>{
  assert.deepEqual(relationEdges('knowledge',[],{modules:['M02','M05','M11']}),[
    {from:'topic',to:'M02',directed:false},{from:'topic',to:'M05',directed:false},{from:'topic',to:'M11',directed:false}
  ]);
  const prereqs=relationEdges('prerequisite',content.prerequisites,{});
  assert.deepEqual(prereqs.map(edge=>[edge.from,edge.to]),content.prerequisites);
  assert.ok(prereqs.every(edge=>edge.directed));
  assert.ok(!prereqs.some(edge=>edge.from==='M13'&&edge.to==='M14'));
  assert.ok(curveBetween({x:0,y:1},{x:100,y:90}).endsWith('100 90'));
});

test('loss chart remains bounded and accurate for falling or constant observations',()=>{
  const points=lossHistoryPoints([2,1,.5]).split(' ').map(pair=>pair.split(',').map(Number));
  assert.deepEqual(points[0],[8,8]);assert.deepEqual(points.at(-1),[332,74]);
  for(const [x,y] of points){assert.ok(x>=8&&x<=332);assert.ok(y>=8&&y<=74);}
  assert.equal(lossHistoryPoints([1,1]),'8,74 332,74');
  assert.equal(lossHistoryPoints([]),'');assert.throws(()=>lossHistoryPoints([NaN]));
});

test('experiment keeps persistent probability targets and meaningful mask feedback',()=>{
  const shell=tokenResultsShell();
  assert.equal([...shell.matchAll(/data-prob-row=/g)].length,3);
  assert.equal([...shell.matchAll(/class="prob-ghost"/g)].length,3);
  assert.ok(shell.includes('id="step-feedback"'));assert.ok(shell.includes('id="loss-history"'));
  const result=batchResultsHtml(batchComparison([2,1,0]),[1,1,1,1,0,0]);
  assert.ok(result.includes('1.907606'));assert.ok(result.includes('1.407606'));assert.ok(result.includes('≠'));
  assert.equal([...result.matchAll(/class="batch-token masked"/g)].length,2);
  assert.ok(batchResultsHtml(batchComparison([2,1,0],[0,0,0,0,0,0]),[0,0,0,0,0,0]).includes('无法定义平均 loss'));
});

test('every bundled font referenced by CSS exists under the published assets',async()=>{
  const css=await fs.readFile(new URL('../../_site/assets/styles.css',import.meta.url),'utf8');
  assert.ok(css.includes('Noto Sans SC Variable'));assert.ok(css.includes('Geist Variable'));
  let count=0;
  for(const match of css.matchAll(/url\((?:"|')?([^)'"\s]+)(?:"|')?\)/g)){
    if(match[1].startsWith('data:'))continue;
    assert.ok(!/^https?:/.test(match[1]));
    await fs.access(new URL('../../_site/assets/'+match[1],import.meta.url));count++;
  }
  assert.ok(count>10);
  const reduced=await fs.readFile(new URL('../modern.css',import.meta.url),'utf8');
  assert.ok(reduced.includes('@media(prefers-reduced-motion:reduce)'));
  for(const name of ['route','network','library','flask','notebook','sun','moon'])assert.ok(icon(name).includes('<svg'));
});

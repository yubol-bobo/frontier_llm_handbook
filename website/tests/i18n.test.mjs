import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {parse} from 'acorn';
import {chooseLocale,getLocale,setLocale,t,ui,messagePattern} from '../i18n.mjs';
import {renderLearningView,escapeText} from '../design.mjs';
import {tokenResultsShell,batchResultsHtml} from '../lab-view.mjs';
import {batchComparison,validateProgress,emptyProgress} from '../learning.mjs';

test('explicit language links override saved preferences and reject unknown locales',()=>{
  assert.equal(chooseLocale('https://example.test/?lang=en#/lab','zh'),'en');
  assert.equal(chooseLocale('https://example.test/?lang=zh#/lab','en'),'zh');
  assert.equal(chooseLocale('https://example.test/#/lab','en'),'en');
  assert.equal(chooseLocale('https://example.test/?lang=invalid','invalid'),'zh');
  assert.throws(()=>setLocale('invalid'));assert.equal(getLocale(),'zh');
});

test('every authored UI phrase has an English translation',async()=>{
  setLocale('en');let checked=0;
  try{
    const visit=node=>{
      if(!node||typeof node!=='object')return;
      const value=node.type==='Literal'&&typeof node.value==='string'?node.value:node.type==='TemplateElement'?node.value.cooked:null;
      if(value)for(const match of value.matchAll(messagePattern())){assert.ok(!/\p{Script=Han}/u.test(t(match[0])),JSON.stringify(match[0]));checked++;}
      for(const value of Object.values(node))if(Array.isArray(value))value.forEach(visit);else if(value&&typeof value==='object')visit(value);
    };
    for(const file of ['app.mjs','design.mjs','interactions.mjs','lab-view.mjs','learning.mjs'])visit(parse(await fs.readFile(new URL('../'+file,import.meta.url),'utf8'),{ecmaVersion:'latest',sourceType:'module'}));
    const shell=await fs.readFile(new URL('../index.html',import.meta.url),'utf8');
    for(const match of shell.matchAll(/data-i18n(?:-[a-z-]+)?="([^"]+)"/g)){assert.ok(!/\p{Script=Han}/u.test(t(match[1])),match[1]);checked++;}
    assert.ok(checked>250);
  }finally{setLocale('zh');}
});

test('translation never rewrites interpolated personal notes, queries, or code identifiers',()=>{
  setLocale('en');
  try{
    const personal='学习路径，loss_mask & <script>private()</script>';
    assert.equal(ui`<p>我的模块记录</p><pre>${escapeText(personal)}</pre>`,'<p>My module notes</p><pre>'+escapeText(personal)+'</pre>');
    assert.equal(ui`搜索「${personal}」`,'Search: “'+personal+'”');
    const record=emptyProgress();record.modules.M00={status:'learning',note:personal,evidence:'原始记录。',updated:''};
    assert.deepEqual(validateProgress(record),record);
  }finally{setLocale('zh');}
});

test('English lab uses correct loss terminology and retains the independent numerical counterexample',()=>{
  setLocale('en');
  try{
    const shell=tokenResultsShell();assert.ok(shell.includes('cross-entropy loss'));assert.ok(!/\p{Script=Han}/u.test(shell));
    const output=batchResultsHtml(batchComparison([2,1,0]),[1,1,1,1,0,0]);
    assert.ok(output.includes('Unweighted mean of local means'));assert.ok(output.includes('1.907606'));assert.ok(output.includes('1.407606'));
    assert.ok(!/\p{Script=Han}/u.test(output));
    assert.ok(batchResultsHtml(batchComparison([2,1,0],[0,0,0,0,0,0]),[0,0,0,0,0,0]).includes('mean loss is undefined'));
  }finally{setLocale('zh');}
});

test('English home uses translated curriculum data without changing module identity or progress',async()=>{
  const content=JSON.parse(await fs.readFile(new URL('../../_site/content.en.json',import.meta.url),'utf8'));
  setLocale('en');
  try{
    const html=renderLearningView({data:content,progress:{lastModule:'M08'},entry:()=>({status:'learning'}),readLink:file=>'#/read/'+file,moduleLink:id=>'#/module/'+id,statusTag:()=>'<span>In progress</span>',stageFilter:'all'});
    assert.equal([...html.matchAll(/class="module-card"/g)].length,16);
    assert.ok(html.includes('<span class="continue-module">M08'));
    assert.ok(html.includes('Continue learning'));
    assert.ok(!/\p{Script=Han}/u.test(html.replace(/<[^>]*>/g,'')));
  }finally{setLocale('zh');}
});

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {renderer,rewriteLink,slug} from '../content.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const content=JSON.parse(await fs.readFile(path.join(root,'_site/content.json'),'utf8'));

test('generated curriculum and source inventory preserve canonical identities',async()=>{
  assert.deepEqual(content.modules.map(m=>m.id),Array.from({length:16},(_,i)=>'M'+String(i).padStart(2,'0')));
  assert.deepEqual(content.seminars.map(s=>s.id),['F01','F02','F03','F04','F05','F06']);
  for(const seminar of content.seminars)assert.ok(!/^F\d+/.test(seminar.title));
  const registry=JSON.parse(await fs.readFile(path.join(root,'repos.json'),'utf8')).repositories;
  assert.deepEqual(content.resources.map(r=>r.id),registry.map(r=>r.id));
  assert.equal(content.resources.length,21);
  assert.deepEqual(content.modules.find(m=>m.id==='M14').prerequisites,['M03']);
  for(const m of content.modules){assert.ok(m.html.includes('验收'));assert.ok(m.html.includes('练习'));}
});

test('every generated internal document route and fragment exists',()=>{
  const errors=[];let checked=0;
  for(const doc of [...Object.values(content.docs),...content.modules]){
    for(const match of doc.html.matchAll(/href="(#\/read\/[^" ]+)"/g)){
      const route=match[1].replaceAll('&amp;','&'),[pathname,query='']=route.slice('#/read/'.length).split('?');
      const file=decodeURIComponent(pathname),target=content.docs[file],anchor=new URLSearchParams(query).get('anchor');checked++;
      if(!target)errors.push(`${doc.path||doc.id}: missing ${file}`);
      else if(anchor&&!target.html.includes('id="'+anchor+'"'))errors.push(`${doc.path||doc.id}: missing ${file}#${anchor}`);
    }
  }
  assert.ok(checked>300);assert.deepEqual(errors,[]);
});

test('local ignored source paths become exact upstream snapshots',()=>{
  const paths=new Set(['ROADMAP.md','curriculum/core.md']);
  const snapshots={example:{url:'https://github.com/org/repo.git',commit:'a'.repeat(40)}};
  assert.equal(rewriteLink('../ROADMAP.md','curriculum/core.md',paths,snapshots),'#/read/ROADMAP.md');
  assert.equal(rewriteLink('#test','curriculum/core.md',paths,snapshots),'#/read/curriculum/core.md?anchor=test');
  assert.equal(rewriteLink('../sources/example/src/a.py#L7','curriculum/core.md',paths,snapshots),'https://github.com/org/repo/blob/'+ 'a'.repeat(40) +'/src/a.py#L7');
  assert.equal(rewriteLink('javascript:alert(1)','ROADMAP.md',paths,snapshots),'#/learn');
  assert.equal(rewriteLink('../../private','ROADMAP.md',paths,snapshots),'#/learn');
  assert.equal(rewriteLink('ROADMAP.md#%','ROADMAP.md',paths,snapshots),'#/learn');
  assert.equal(rewriteLink('sources/example','ROADMAP.md',paths,snapshots),'https://github.com/org/repo/tree/'+'a'.repeat(40));
});

test('Markdown preserves tables, source anchors and fenced code while excluding executable HTML',()=>{
  const markdown='# 标题\n\n<a id="m01"></a>\n\n<script>alert(1)</script><img src="x" onerror="alert(1)">\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n```python\nprint("a < b")\n```\n';
  const result=renderer('sample.md',new Set(['sample.md']),{}).render(markdown);
  assert.ok(result.html.includes('<table>'));assert.ok(result.html.includes('id="m01"'));assert.ok(result.html.includes('language-python'));
  assert.ok(!result.html.includes('<script'));assert.ok(!result.html.includes('onerror='));assert.equal(slug('知识与系统：入门'),'知识与系统入门');
});

test('source evidence, manuscript text and connection targets are retained',async()=>{
  assert.equal(content.connections.length,8);
  for(const c of content.connections){assert.ok(content.docs[c.source]);for(const id of c.modules)assert.ok(content.modules.some(m=>m.id===id));}
  for(const r of content.resources){assert.match(r.commit,/^[a-f0-9]{40}$/);assert.ok(content.docs[r.note].markdown.includes(r.commit));assert.ok(r.question&&r.stop);}
  const raw=await fs.readFile(path.join(root,'lessons/01-one-token-to-update.md'),'utf8');
  assert.equal(content.docs['lessons/01-one-token-to-update.md'].markdown,raw.replace(/\r\n/g,'\n'));
});

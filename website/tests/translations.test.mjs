import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import MarkdownIt from 'markdown-it';

const zh=JSON.parse(await fs.readFile(new URL('../../_site/content.json',import.meta.url),'utf8'));
const en=JSON.parse(await fs.readFile(new URL('../../_site/content.en.json',import.meta.url),'utf8'));
const links=html=>[...html.matchAll(/href="([^"]+)"/g)].map(match=>match[1]).sort();
const md=new MarkdownIt({html:true});

test('all 51 English documents retain canonical identity, references, links and legacy fragments',()=>{
  assert.equal(zh.locale,'zh');assert.equal(en.locale,'en');
  assert.deepEqual(Object.keys(en.docs),Object.keys(zh.docs));assert.equal(Object.keys(en.docs).length,51);
  assert.deepEqual(en.modules.map(m=>[m.id,m.file,m.anchor,m.prerequisites]),zh.modules.map(m=>[m.id,m.file,m.anchor,m.prerequisites]));
  assert.deepEqual(en.seminars.map(m=>[m.id,m.file,m.anchor]),zh.seminars.map(m=>[m.id,m.file,m.anchor]));
  assert.deepEqual(en.resources.map(r=>[r.id,r.commit,r.firstModule]),zh.resources.map(r=>[r.id,r.commit,r.firstModule]));
  assert.deepEqual(en.prerequisites,zh.prerequisites);
  assert.deepEqual(en.connections.map(c=>[c.id,c.modules,c.source,c.lab]),zh.connections.map(c=>[c.id,c.modules,c.source,c.lab]));
  for(const [file,doc] of Object.entries(en.docs)){
    assert.deepEqual(links(doc.html),links(zh.docs[file].html),'References changed: '+file);
    for(const match of zh.docs[file].html.matchAll(/\bid="([^"]+)"/g))assert.ok(doc.html.includes('id="'+match[1]+'"'),file+' lost fragment '+match[1]);
  }
  let checked=0;
  for(const doc of [...Object.values(en.docs),...en.modules])for(const match of doc.html.matchAll(/href="(#\/read\/[^" ]+)"/g)){
    const [file,query='']=match[1].replaceAll('&amp;','&').slice('#/read/'.length).split('?'),target=en.docs[decodeURIComponent(file)],anchor=new URLSearchParams(query).get('anchor');
    assert.ok(target,'Missing English route '+file);if(anchor)assert.ok(target.html.includes('id="'+anchor+'"'),'Missing English fragment '+file+'#'+anchor);checked++;
  }
  assert.ok(checked>300);
});

test('English translation preserves exercises, tables, executable code and numerical examples',()=>{
  const shape=text=>{
    const tokens=md.parse(text,{});
    return {
      headings:tokens.filter(t=>t.type==='heading_open').map(t=>t.tag),
      exercises:tokens.filter(t=>t.type==='list_item_open').length,
      cells:tokens.filter(t=>['td_open','th_open'].includes(t.type)).length,
      code:tokens.filter(t=>t.type==='fence'&&!['text','plaintext','mermaid',''].includes(t.info)).map(t=>[t.info,t.content])
    };
  };
  for(const [file,doc] of Object.entries(en.docs)){
    assert.deepEqual(shape(doc.markdown),shape(zh.docs[file].markdown),file);
    // The glossary intentionally includes a Chinese column; source code is preserved verbatim.
    if(file!=='GLOSSARY.md')assert.ok(!/\p{Script=Han}/u.test(doc.html.replace(/<pre\b[^>]*>[\s\S]*?<\/pre>/g,'').replace(/<code\b[^>]*>[\s\S]*?<\/code>/g,'').replace(/<[^>]*>/g,'')),'Untranslated prose: '+file);
  }
  for(const term of ['model FLOPs utilization','hardware FLOPs utilization','context parallelism','sequence parallelism','fully sharded data parallelism','hybrid sharded data parallelism','group relative policy optimization','direct preference optimization'])assert.ok(en.docs['GLOSSARY.md'].markdown.includes(term),term);
  const lesson=en.docs['lessons/01-one-token-to-update.md'].markdown;
  const decimals=zh.docs['lessons/01-one-token-to-update.md'].markdown.match(/\b\d+\.\d{3,}\b/g);
  assert.ok(decimals.length>5);
  for(const value of decimals)assert.ok(lesson.includes(value),value);
});

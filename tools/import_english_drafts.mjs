// Import reviewed drafts while preserving canonical links and legacy heading anchors.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import MarkdownIt from 'markdown-it';
import {buildContent,renderer,rewriteLink} from '../website/content.mjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const folders=process.argv.slice(2).map(p=>path.resolve(p));
if(!folders.length)throw new Error('Pass one or more reviewed draft directories.');
const content=await buildContent(root),files=new Set(Object.keys(content.docs));
const registry=JSON.parse(await fs.readFile(path.join(root,'repos.json'),'utf8')).repositories;
const lock=JSON.parse(await fs.readFile(path.join(root,'sources.lock.json'),'utf8')).repositories;
const snapshots=Object.fromEntries(lock.map(r=>[r.id,{...registry.find(x=>x.id===r.id),...r}]));
const md=new MarkdownIt({html:true,linkify:true}),pending=[],errors=[];
const signature=markdown=>{
  const tokens=md.parse(markdown,{}),inline=tokens.flatMap(t=>t.children||[]);
  return {
    headings:tokens.filter(t=>t.type==='heading_open').map(t=>t.tag),
    links:inline.filter(t=>t.type==='link_open').map(t=>t.attrGet('href')).sort(),
    code:tokens.filter(t=>t.type==='fence'&&!['mermaid','text','plaintext',''].includes(t.info.trim())).map(t=>[t.info,t.content]),
    listItems:tokens.filter(t=>t.type==='list_item_open').length,
    tableCells:tokens.filter(t=>['th_open','td_open'].includes(t.type)).length
  };
};
for(const file of files){
  let translated;
  for(const folder of folders){try{translated=await fs.readFile(path.join(folder,file),'utf8');break;}catch(e){if(e.code!=='ENOENT')throw e;}}
  if(translated===undefined){errors.push('Missing draft: '+file);continue;}
  translated=translated.replace(/\r\n/g,'\n');
  const original=content.docs[file].markdown,sourceShape=signature(original),targetShape=signature(translated);
  for(const key of Object.keys(sourceShape))if(JSON.stringify(sourceShape[key])!==JSON.stringify(targetShape[key]))errors.push(file+': changed '+key);
  const codeSpans=text=>md.parse(text,{}).flatMap(token=>token.children||[]).filter(token=>token.type==='code_inline').map(token=>token.content);
  const before=codeSpans(original),after=codeSpans(translated);
  for(const code of new Set(before.filter(code=>!/\p{Script=Han}/u.test(code))))if(after.filter(value=>value===code).length<before.filter(value=>value===code).length)errors.push(file+': missing code identifier '+code);
  const translatedHeadings=renderer(file,files,snapshots).render(translated).headings;
  const lines=translated.split('\n'),tokens=md.parse(translated,{}).filter(token=>token.type==='heading_open');
  // Keep Chinese fragment URLs usable both on the website and in GitHub's Markdown view.
  for(let i=tokens.length-1;i>=0;i--){
    const originalId=content.docs[file].headings[i]?.id;
    if(originalId&&originalId!==translatedHeadings[i]?.id&&!translated.includes('id="'+originalId+'"'))lines.splice(tokens[i].map[0],0,'<a id="'+originalId+'"></a>','');
  }
  translated=lines.join('\n').replace(/(\[[^\]\n]*\]\()([^\)\n]+)(\))/g,(all,start,href,end)=>{
    if(/^(?:https?:|mailto:|#)/.test(href))return all;
    const target=path.posix.normalize(path.posix.join(path.posix.dirname(file),decodeURIComponent(href.split('#')[0])));
    // Translated documents resolve within translations/en. Nontranslated assets use canonical URLs.
    return files.has(target)?all:start+rewriteLink(href,file,files,snapshots)+end;
  });
  pending.push({file,translated,hash:createHash('sha256').update(original).digest('hex')});
}
if(errors.length){console.error(JSON.stringify(errors,null,2));process.exit(1);}
const manifest={schemaVersion:1,sourceLocale:'zh',targetLocale:'en',files:{}};
for(const item of pending){
  const destination=path.join(root,'translations/en',item.file);
  await fs.mkdir(path.dirname(destination),{recursive:true});await fs.writeFile(destination,item.translated);
  manifest.files[item.file]=item.hash;
}
await fs.writeFile(path.join(root,'translations/en/manifest.json'),JSON.stringify(manifest,null,2)+'\n');
console.log('Imported '+pending.length+' English documents with source hashes and canonical heading aliases.');

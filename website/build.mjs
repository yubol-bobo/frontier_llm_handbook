import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { build } from 'esbuild';
import { buildContent } from './content.mjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const out=path.join(root,'_site');
await fs.mkdir(path.join(out,'assets'),{recursive:true});
const content=await buildContent(root);
content.revision=execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim();
await fs.writeFile(path.join(out,'content.json'),JSON.stringify(content));
const englishContent=await buildContent(root,'en');
englishContent.revision=content.revision;
await fs.writeFile(path.join(out,'content.en.json'),JSON.stringify(englishContent));
await fs.copyFile(path.join(root,'website/index.html'),path.join(out,'index.html'));
await build({entryPoints:[path.join(root,'website/styles.css')],bundle:true,minify:true,outfile:path.join(out,'assets/styles.css'),loader:{'.woff2':'file','.woff':'file'},assetNames:'fonts/[name]-[hash]'});
await fs.copyFile(path.join(root,'website/favicon.svg'),path.join(out,'assets/favicon.svg'));
await fs.mkdir(path.join(out,'assets/licenses'),{recursive:true});
for(const [packageName,file] of [['@fontsource-variable/geist','geist.txt'],['@fontsource-variable/noto-sans-sc','noto-sans-sc.txt'],['lucide','lucide.txt']]){
  await fs.copyFile(path.join(root,'node_modules',packageName,'LICENSE'),path.join(out,'assets/licenses',file));
}
await fs.writeFile(path.join(out,'.nojekyll'),'');
await build({entryPoints:[path.join(root,'website/app.mjs')],bundle:true,splitting:true,format:'esm',outdir:path.join(out,'assets'),entryNames:'app',chunkNames:'chunks/[name]-[hash]',minify:true,target:['es2022'],legalComments:'linked'});
console.log(JSON.stringify({output:'_site',documents:Object.keys(content.docs).length,modules:content.modules.length,seminars:content.seminars.length,repositories:content.resources.length,revision:content.revision},null,2));

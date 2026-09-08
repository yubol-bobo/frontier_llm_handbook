import fs from 'node:fs/promises';
import path from 'node:path';
import MarkdownIt from 'markdown-it';
import sanitizeHtml from 'sanitize-html';
import {createHash} from 'node:crypto';
import englishMetadata from './locales/en-content.json' with {type:'json'};

export const repoUrl = 'https://github.com/yubol-bobo/frontier_llm_handbook';
export const stages = [
  { id: 'foundations', name: '基础与预训练', range: 'M00—M04', description: '从一枚 token，走到一份可信的训练配方。', ids: ['M00','M01','M02','M03','M04'] },
  { id: 'systems', name: '训练与推理系统', range: 'M05—M09', description: '理解张量如何分布、计算，以及可靠地恢复。', ids: ['M05','M06','M07','M08','M09'] },
  { id: 'agents', name: '后训练与 Agent', range: 'M10—M15', description: '将反馈变成更新，用独立评估验证实际能力。', ids: ['M10','M11','M12','M13','M14','M15'] }
];

export const connections = [
  { id:'loss', title:'同一个 loss 分母，贯穿数据、分布式与 RL', modules:['M02','M03','M05','M11','M13'], text:'Packing 和 mask 改变有效 token 数。各分片有效长度不同时，直接平均局部均值会改变全局 token 目标；RL 轨迹也需要明确加权单位。', evidence:'分组加权反例已有 CPU 实验；真实分布式与 RL 运行尚待验证。', source:'notes/connections/end-to-end.md', lab:true },
  { id:'data', title:'数据身份，决定实验能否解释', modules:['M03','M04','M08'], text:'数据来源、tokenizer、处理版本、packing 和实际混合比例需要跟随运行保存，才能判断改进来自配方、预算还是数据改变。', evidence:'公开配方与数据产物契约的综合对应。', source:'handbook/01-data-model-pretraining-design.md' },
  { id:'recovery', title:'张量放在哪里，决定怎样恢复训练', modules:['M05','M08','M09'], text:'并行布局决定参数与优化器状态的所有权。恢复还需要训练配置要求的 master 权重、随机状态、数据位置与完整提交条件。', evidence:'固定源码与公开案例综合，未验证不同拓扑恢复逐位一致。', source:'handbook/02-distributed-pretraining-operations.md' },
  { id:'moe', title:'MoE 通信布局，也是内核的输入契约', modules:['M06','M07','M09'], text:'路由将 token 整理为按 expert 对齐的数据行，内核消费布局，逆路由合回原 token。元数据、stream 与 buffer 生命周期共同影响正确性和性能。', evidence:'存在具体 adapter 的源码证据；不同 DeepEP 接口版本不能直接混用。', source:'notes/connections/infra.md' },
  { id:'reward', title:'任务反馈，怎样成为训练目标', modules:['M11','M12','M13','M14'], text:'环境反馈需要明确的奖励维度与标量化规则。轨迹同时保留动作 token、logprob、mask、分支及版本，才能进入正确的更新。', evidence:'CPU 实验已检查 Python 奖励断言；Harbor 奖励加载与完整训练集成待验证。', source:'handbook/03-posttraining-agent-rl-evaluation.md' },
  { id:'policy', title:'生成吞吐与策略一致性，需要一起设计', modules:['M07','M09','M12','M13'], text:'权重交接需要完整策略版本，并对齐 tokenizer、template、路由、精度和采样。策略滞后、数值差异与旧 KV 复用需要分别诊断。', evidence:'跨层语义综合；独立源码快照不是已验证的兼容栈。', source:'notes/connections/end-to-end.md' },
  { id:'context', title:'上下文变长，整条链路都要重新验收', modules:['M03','M06','M08','M09','M13'], text:'长上下文改变样本组织、专家负载、显存与通信，继而影响 rollout 耗时、KV cache、队列陈旧度及恢复成本。', evidence:'多个公开实现的互补案例，不能把一处阈值推广到所有系统。', source:'notes/connections/end-to-end.md' }
];

export function slug(text) {
  return text.toLowerCase().trim().replace(/<[^>]+>/g, '').replace(/[^\p{L}\p{N}\p{M}\s_-]/gu, '').replace(/\s/g, '-');
}

export function routeForDoc(file, anchor='') {
  return '#/read/' + file.split('/').map(encodeURIComponent).join('/') + (anchor ? '?anchor=' + encodeURIComponent(anchor) : '');
}

export function rewriteLink(href, current, documentPaths, snapshots) {
  if (/^(https?:|mailto:)/i.test(href)) return href;
  if (/^[a-z][a-z\d+.-]*:/i.test(href) || href.startsWith('//')) return '#/learn';
  const [rawPath, rawAnchor=''] = href.split('#');
  let target,anchor;
  try { target = path.posix.normalize(path.posix.join(path.posix.dirname(current), decodeURIComponent(rawPath || path.posix.basename(current)))); anchor = decodeURIComponent(rawAnchor); }
  catch { return '#/learn'; }
  if (documentPaths.has(target)) return routeForDoc(target, anchor);
  const parts = target.split('/').filter(Boolean);
  if (parts[0] === 'sources' && snapshots[parts[1]]) {
    const source = snapshots[parts[1]];
    return source.url.replace(/\.git$/, '') + (parts.length<=2?'/tree/':'/blob/') + source.commit + (parts.length>2?'/'+parts.slice(2).map(encodeURIComponent).join('/'):'') + (anchor ? '#' + anchor : '');
  }
  if (target.startsWith('../') || path.posix.isAbsolute(target)) return '#/learn';
  return repoUrl + '/blob/main/' + target.split('/').map(encodeURIComponent).join('/') + (anchor ? '#' + anchor : '');
}

export function renderer(file, documentPaths, snapshots) {
  const md = new MarkdownIt({ html:true, linkify:true, typographer:false });
  const headingIds = new Map();
  const headings=[];
  md.renderer.rules.heading_open = (tokens, index) => {
    const label = tokens[index+1].children?.filter(t => t.type==='text' || t.type==='code_inline').map(t=>t.content).join('') || tokens[index+1].content;
    const base=slug(label), count=headingIds.get(base)||0;
    headingIds.set(base,count+1);
    const id=base+(count ? '-'+count : '');
    headings.push({id,label,level:Number(tokens[index].tag.slice(1))});
    return '<'+tokens[index].tag+' id="'+md.utils.escapeHtml(id)+'">';
  };
  const originalLink=md.renderer.rules.link_open || ((tokens,i,o,e,s)=>s.renderToken(tokens,i,o));
  md.renderer.rules.link_open = (tokens,i,o,e,s) => {
    const href=rewriteLink(tokens[i].attrGet('href')||'',file,documentPaths,snapshots);
    tokens[i].attrSet('href',href);
    if (/^https?:/.test(href)) { tokens[i].attrSet('target','_blank'); tokens[i].attrSet('rel','noopener noreferrer'); }
    return originalLink(tokens,i,o,e,s);
  };
  const fence=md.renderer.rules.fence;
  md.renderer.rules.fence=(tokens,i,o,e,s)=> tokens[i].info.trim()==='mermaid'
    ? '<pre class="mermaid">'+md.utils.escapeHtml(tokens[i].content)+'</pre>'
    : fence(tokens,i,o,e,s);
  return {
    render(text) {
      const html=sanitizeHtml(md.render(text), {
        allowedTags:sanitizeHtml.defaults.allowedTags.concat(['details','summary','input','img']),
        allowedAttributes:{...sanitizeHtml.defaults.allowedAttributes,'*':['id','class'],a:['href','target','rel','id'],img:['src','alt','title'],input:['type','checked','disabled'],th:['align'],td:['align']},
        allowedSchemes:['http','https','mailto'], allowProtocolRelative:false
      });
      return {html,headings};
    }
  };
}

async function walk(root, relative) {
  const result=[];
  for (const entry of await fs.readdir(path.join(root,relative),{withFileTypes:true})) {
    const file=relative+'/'+entry.name;
    if(entry.isDirectory()) result.push(...await walk(root,file));
    else if(entry.name.endsWith('.md')) result.push(file);
  }
  return result.sort();
}

export async function buildContent(root,locale='zh') {
  if(!['zh','en'].includes(locale))throw new Error('Unsupported content locale');
  const roots=['README.md','ROADMAP.md','RESOURCE_ATLAS.md','COVERAGE.md','KNOWLEDGE_TREE.md','REPO_RELATIONSHIPS.md','LEARNING_LIST.md','HOW_TO_STUDY.md','GLOSSARY.md','CONTRIBUTING.md','SOURCE_INDEX.md','PROGRESS.md'];
  const files=[...roots];
  for(const folder of ['curriculum','handbook','lessons','notes/repositories','notes/connections','templates','experiments']) files.push(...await walk(root,folder));
  const documentPaths=new Set(files);
  const registry=JSON.parse(await fs.readFile(path.join(root,'repos.json'),'utf8')).repositories;
  const lock=JSON.parse(await fs.readFile(path.join(root,'sources.lock.json'),'utf8')).repositories;
  const snapshots=Object.fromEntries(lock.map(r=>[r.id,{...registry.find(x=>x.id===r.id),...r}]));
  const docs={};
  const manifest=locale==='en'?JSON.parse(await fs.readFile(path.join(root,'translations/en/manifest.json'),'utf8')):null;
  for(const file of files) {
    const original=(await fs.readFile(path.join(root,file),'utf8')).replace(/\r\n/g,'\n');
    if(manifest&&manifest.files[file]!==createHash('sha256').update(original).digest('hex'))throw new Error('English translation needs a source review: '+file);
    const markdown=locale==='en'?(await fs.readFile(path.join(root,'translations/en',file),'utf8')).replace(/\r\n/g,'\n'):original;
    const title=markdown.match(/^# (.+)$/m)?.[1]||file;
    const {html,headings}=renderer(file,documentPaths,snapshots).render(markdown);
    const minutes=locale==='en'?Math.ceil(markdown.split(/\s+/).length/220):Math.ceil(markdown.length/650);
    docs[file]={path:file,title,html,headings,markdown,minutes:Math.max(1,minutes)};
  }
  const modules=[];
  for(const match of docs['ROADMAP.md'].markdown.matchAll(/^\| \[(M\d+) (.*?)\]\(([^#]+)#(m\d+)\) \| (.*?) \| (.*?) \| (.*?) \|$/gm)) {
    const [,id,title,file,anchor,question,materials,deliverable]=match;
    const section=docs[file].markdown.split('<a id="'+anchor+'"></a>')[1]?.split(/<a id="[mf]\d+"><\/a>/)[0]?.trim();
    if(!section) throw new Error('Missing module section '+id);
    const rendered=renderer(file,documentPaths,snapshots).render(section);
    modules.push({id,title,file,anchor,question,materials,deliverable,html:rendered.html,headings:rendered.headings,stage:stages.find(s=>s.ids.includes(id)).id});
  }
  if(modules.length!==16) throw new Error('Expected 16 core modules, found '+modules.length);
  const seminarFile='curriculum/04-frontier-seminars.md';
  const seminars=[...docs[seminarFile].markdown.matchAll(/<a id="(f\d+)"><\/a>\s*(?:<a id="[^"]+"><\/a>\s*)*## ([^\n]+)/g)].map(m=>({id:m[1].toUpperCase(),title:m[2].replace(/^F\d+\s*[｜|]?\s*/,''),file:seminarFile,anchor:m[1]}));
  if(seminars.length!==6) throw new Error('Expected 6 seminars');
  const atlas=docs['RESOURCE_ATLAS.md'].markdown;
  const resources=registry.map(repo=>{
    const row=atlas.split('\n').find(line=>line.includes('(notes/repositories/'+repo.id+'.md)'));
    if(!row)throw new Error('Resource atlas row missing '+repo.id);
    const cells=row.split(' | ');
    return {...repo,url:repo.url.replace(/\.git$/,''),firstModule:cells[1].match(/M\d+/)[0],entry:cells[1],question:cells[2],stop:cells[3].replace(/ \|$/,''),commit:snapshots[repo.id].commit,note:'notes/repositories/'+repo.id+'.md'};
  });
  const prerequisites=[...docs['ROADMAP.md'].markdown.matchAll(/^\s+(M\d+)(?:\[.*?\])? --> (M\d+)/gm)].map(m=>[m[1],m[2]]);
  for(const module of modules) module.prerequisites=prerequisites.filter(e=>e[1]===module.id).map(e=>e[0]);
  for(const c of connections) if(!docs[c.source])throw new Error('Missing connection source '+c.source);
  const stageData=locale==='en'?stages.map(stage=>({...stage,...englishMetadata.stages[stage.id]})):stages;
  const connectionData=locale==='en'?connections.map(connection=>({...connection,...englishMetadata.connections[connection.id]})):connections;
  return {schemaVersion:1,locale,repoUrl,stages:stageData,modules,seminars,resources,connections:connectionData,prerequisites,docs};
}

import {tokenStep,batchComparison,statuses,emptyProgress,validateProgress,searchDocuments} from './learning.mjs';

const main=document.getElementById('main-content');
const storageKey='frontier-llm-handbook:learning:v1';
let data,progress=emptyProgress(),storageAvailable=true,routeVersion=0,toastTimer;
let stageFilter='all',resourceFilter='all',mapMode='knowledge',connectionId='loss',selectedModule='M02';
let logits=[2,1,0],target=1,rate=0.1,masks=[1,1,1,1,0,0],partition='uneven',lastStep=null;
const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const readLink=(file,anchor='')=>'#/read/'+file.split('/').map(encodeURIComponent).join('/')+(anchor?'?anchor='+encodeURIComponent(anchor):'');
const moduleLink=id=>'#/module/'+id;
const entry=id=>progress.modules[id]||{status:'new',note:'',evidence:'',updated:''};
const moduleById=id=>data.modules.find(m=>m.id===id);
const titleNames={learn:'学习路径',module:'课程阅读',read:'资料阅读',map:'知识地图',resources:'资源与手册',lab:'互动实验',progress:'我的学习',search:'搜索'};
const categoryLabels={training:'模型与训练',infra:'基础设施',harness:'Agent Harness',rl:'强化学习与环境'};

function toast(message){const element=document.getElementById('toast');element.textContent=message;element.hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>element.hidden=true,4200);}
function saveProgress(){
  try{localStorage.setItem(storageKey,JSON.stringify(progress));storageAvailable=true;return true;}
  catch{storageAvailable=false;toast('浏览器未能保存记录；本次仍可学习，请及时导出。');return false;}
}
function updateSidebar(){
  document.querySelectorAll('[data-status-module]').forEach(element=>{
    const status=entry(element.dataset.statusModule).status;
    element.textContent=statuses[status];
    element.className='tag '+(status==='reviewed'?'green':status==='learning'?'blue':'');
  });
  const done=data.modules.filter(m=>entry(m.id).status==='reviewed').length;
  document.getElementById('side-progress').innerHTML=`<div class="progress-meta"><span>我的核心课程</span><span class="mono">${done} / 16</span></div><div class="progress-track" role="progressbar" aria-label="自主验收进度" aria-valuemin="0" aria-valuemax="16" aria-valuenow="${done}"><div class="progress-fill" style="width:${done/16*100}%"></div></div><small>自主验收 · 仅保存在当前浏览器</small>`;
}
function heading(eyebrow,title,text='',action=''){
  return `<div class="page-heading"><div><div class="eyebrow">${escape(eyebrow)}</div><h1>${escape(title)}</h1>${text?`<p>${escape(text)}</p>`:''}</div>${action}</div>`;
}
function statusTag(id){const status=entry(id).status;return `<span data-status-module="${id}" class="tag ${status==='reviewed'?'green':status==='learning'?'blue':''}">${statuses[status]}</span>`;}
function moduleRow(m){return `<a class="module-row" href="${moduleLink(m.id)}"><span class="module-code">${m.id}</span><div class="module-main"><strong>${escape(m.title)}</strong><p>${escape(m.question)}</p></div><span class="module-materials">${escape(m.materials)}</span>${statusTag(m.id)}<span class="module-arrow" aria-hidden="true">→</span></a>`;}

function renderLearn(){
  const started=progress.lastModule&&entry(progress.lastModule).status!=='reviewed'?moduleById(progress.lastModule):null;
  const next=started||data.modules.find(m=>entry(m.id).status!=='reviewed')||data.modules[0];
  main.innerHTML=heading('Learning path / 公开课程','从训练原理，走到系统实现。','系统学习前沿 LLM 的端到端训练知识，在实现、验证与排障中建立工程能力。',`<a class="button" href="${readLink('ROADMAP.md')}">完整路线与入门诊断 ↗</a>`)+
  `<div class="learning-meta"><span><strong>16</strong> 个核心模块</span><span><strong>6</strong> 个前沿专题</span><span><strong>19</strong> 个固定源码项目</span><span>普通电脑即可开始</span></div>
  <section class="learning-banner" aria-label="下一步学习"><div><div class="eyebrow">${started?'继续上次的学习':'从这里开始'}</div><h2><span class="banner-id">${next.id}</span>${escape(next.title)}</h2><p>${escape(next.deliverable)}</p></div><a class="button" href="${moduleLink(next.id)}">${started?'继续学习':'进入模块'} →</a></section>
  <div class="filter-bar" aria-label="按阶段浏览"><button class="chip ${stageFilter==='all'?'active':''}" data-stage="all" aria-pressed="${stageFilter==='all'}">全部阶段</button>${data.stages.map(s=>`<button class="chip ${stageFilter===s.id?'active':''}" data-stage="${s.id}" aria-pressed="${stageFilter===s.id}">${s.name}</button>`).join('')}<a class="button ghost small" href="#/lab">先动手：一枚 token 的更新 ↗</a></div>
  ${data.stages.filter(s=>stageFilter==='all'||s.id===stageFilter).map((s)=>`<section class="stage"><div class="stage-heading"><span class="stage-number">0${data.stages.indexOf(s)+1}</span><h2>${s.name}</h2><small>${s.range}</small></div><p class="stage-description">${s.description}</p><div class="module-list">${data.modules.filter(m=>m.stage===s.id).map(moduleRow).join('')}</div></section>`).join('')}
  <div class="notice neutral">每个模块包含先修、阅读、练习和验收要求。评估从数据阶段开始贯穿全程；<a href="${readLink('ROADMAP.md')}">已有经验可按诊断进入专项路线</a>。</div>
  <div class="section-title"><h2>走通主干，再探索前沿</h2><span class="tag">按兴趣选读</span></div><div class="seminar-grid">${data.seminars.map(s=>`<a class="seminar" href="${readLink(s.file,s.anchor)}"><span>${s.id}</span>${escape(s.title)} <span aria-hidden="true">↗</span></a>`).join('')}</div>`;
}

function notePanel(id){const saved=entry(id);return `<details class="note-panel" open><summary>我的模块记录</summary><small>笔记仅存在当前浏览器。可在“我的学习”导出备份。</small><label for="module-status">学习状态</label><select id="module-status" data-module="${id}">${Object.entries(statuses).map(([value,label])=>`<option value="${value}" ${saved.status===value?'selected':''}>${label}</option>`).join('')}</select><label for="module-note">理解、疑问与工程记录</label><textarea id="module-note" data-module="${id}" maxlength="20000" placeholder="我能解释什么？实现或排障时发现了什么？">${escape(saved.note)}</textarea><label for="module-evidence">验收证据</label><textarea class="evidence" id="module-evidence" data-module="${id}" maxlength="5000" placeholder="代码或笔记链接、验证结果与未完成部分">${escape(saved.evidence)}</textarea><small>“已自主验收”需填写证据，并自行核对模块出口要求。</small><button class="button small" data-action="save-module" data-id="${id}">保存状态与记录</button><div id="save-feedback" class="save-feedback" role="status"></div></details>`;}
function toc(headings){return `<div><h3>本页目录</h3><nav class="toc" aria-label="本页目录">${headings.filter(h=>h.level<=3).map(h=>`<button data-heading="${escape(h.id)}">${escape(h.label)}</button>`).join('')}</nav></div>`;}
function readerTools(){
  main.querySelectorAll('.markdown pre:not(.mermaid)').forEach(pre=>{
    const toolbar=document.createElement('div');toolbar.className='code-toolbar';
    const button=document.createElement('button');button.type='button';button.textContent='复制';button.setAttribute('aria-label','复制代码块');
    button.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(pre.textContent);button.textContent='已复制';setTimeout(()=>button.textContent='复制',1800);}catch{toast('无法自动复制，请选中代码复制。');}});
    toolbar.append(button);pre.before(toolbar);
  });
}
async function renderMermaid(version){
  const nodes=[...main.querySelectorAll('pre.mermaid')];if(!nodes.length)return;
  try{
    const {default:mermaid}=await import('mermaid');
    if(version!==routeVersion)return;
    mermaid.initialize({startOnLoad:false,securityLevel:'strict',theme:'base',themeVariables:{primaryColor:'#eaf2ff',primaryTextColor:'#182d42',primaryBorderColor:'#8bb2de',lineColor:'#6d8cad',fontFamily:'Segoe UI, Microsoft YaHei, sans-serif',fontSize:'16px'},flowchart:{htmlLabels:false,useMaxWidth:true},suppressErrorRendering:true});
    for(let i=0;i<nodes.length;i++){
      if(version!==routeVersion)return;
      const source=nodes[i].textContent;
      try{
        const {svg}=await mermaid.render('diagram-'+version+'-'+i,source);
        if(version!==routeVersion)return;
        const diagram=document.createElement('div');diagram.className='mermaid';diagram.innerHTML=svg;diagram.setAttribute('role','img');diagram.setAttribute('aria-label','文档关系图，展开下方文字可阅读图的原始关系');
        const details=document.createElement('details');const summary=document.createElement('summary');summary.textContent='查看图的文字关系';const pre=document.createElement('pre');pre.textContent=source;details.append(summary,pre);nodes[i].replaceWith(diagram,details);
      }catch{nodes[i].classList.remove('mermaid');nodes[i].setAttribute('aria-label','关系图文字源，图形暂无法显示');}
    }
  }catch{toast('关系图暂时无法加载，已保留文字关系。');}
}
function renderModule(id){
  const m=moduleById(id);if(!m)return renderMissing();
  progress.lastModule=id;saveProgress();
  const index=data.modules.indexOf(m),stage=data.stages.find(s=>s.id===m.stage);
  main.innerHTML=`<div class="reader-meta"><a href="#/learn">← 学习路径</a><span class="tag">${stage.name}</span><span class="tag blue">${id}</span>${statusTag(id)}</div><div class="reader-heading"><h1>${escape(m.title)}</h1><p>${escape(m.question)}</p><div class="prerequisites"><span class="muted">先修：</span>${m.prerequisites.length?m.prerequisites.map(p=>`<a href="${moduleLink(p)}">${p} ${escape(moduleById(p).title)}</a>`).join(''):'Python 基础；按模块诊断补齐'}<a class="source-link" href="${data.repoUrl}/blob/main/${m.file}#${m.anchor}" target="_blank" rel="noopener noreferrer">原始课程 ↗</a></div></div>
  <div class="reader-layout"><div class="reader-body"><div class="notice neutral">这是课程要求与练习设计；完成阅读后，请通过实现、解释和证据验收自己的能力。<a href="${readLink('COVERAGE.md')}">查看教材与实验完成范围</a>。</div><article class="markdown">${m.html.replace(/^<h2[^>]*>.*?<\/h2>\s*/s,'')}</article><nav class="reading-actions" aria-label="模块前后导航">${index?`<a class="button" href="${moduleLink(data.modules[index-1].id)}">← ${data.modules[index-1].id} 上一模块</a>`:'<a class="button" href="#/lab">体验互动实验</a>'}${index<15?`<a class="button primary" href="${moduleLink(data.modules[index+1].id)}">${data.modules[index+1].id} 下一模块 →</a>`:'<a class="button primary" href="#/progress">整理综合项目 →</a>'}</nav></div><aside class="reader-aside">${toc(m.headings.slice(1))}${notePanel(id)}</aside></div>`;
  readerTools();
}
function renderDocument(file,anchor){
  const doc=Object.hasOwn(data.docs,file)?data.docs[file]:null;if(!doc)return renderMissing();
  main.innerHTML=`<div class="reader-meta"><a href="#/resources">← 资源与手册</a><span class="tag">约 ${doc.minutes} 分钟阅读 · 不含练习</span><a class="source-link" href="${data.repoUrl}/blob/main/${encodeURI(file)}${anchor?'#'+encodeURIComponent(anchor):''}" target="_blank" rel="noopener noreferrer">在 GitHub 查看原文 ↗</a></div><div class="reader-layout"><div class="reader-body"><article class="markdown">${doc.html}</article></div><aside class="reader-aside">${toc(doc.headings)}<div class="notice neutral">内容来自资料库原文。<a href="#/learn">回到学习路径</a>，可按模块保存笔记与验收记录。</div></aside></div>`;
  readerTools();
  if(anchor)requestAnimationFrame(()=>scrollHeading(anchor));
}
function scrollHeading(id){const element=document.getElementById(id);if(element&&main.contains(element))element.scrollIntoView({block:'start',behavior:'instant'});}

function renderMap(){
  const selected=data.connections.find(c=>c.id===connectionId)||data.connections[0];
  const m=moduleById(selectedModule)||data.modules[2];
  const upstream=m.prerequisites,downstream=data.prerequisites.filter(p=>p[0]===m.id).map(p=>p[1]);
  const highlights=mapMode==='knowledge'?selected.modules:[m.id,...upstream,...downstream];
  main.innerHTML=heading('Knowledge map / 跨层理解','把知识连起来。','选择一个问题，观察它怎样跨越课程阶段，再沿证据回到实际实现。')+
  `<div class="filter-bar"><button class="chip ${mapMode==='knowledge'?'active':''}" data-map-mode="knowledge" aria-pressed="${mapMode==='knowledge'}">跨模块知识联系</button><button class="chip ${mapMode==='prerequisite'?'active':''}" data-map-mode="prerequisite" aria-pressed="${mapMode==='prerequisite'}">课程先修关系</button><a class="button ghost small" href="${readLink('REPO_RELATIONSHIPS.md')}">查看真实软件依赖 ↗</a></div><div class="map-layout"><section class="map-board" aria-label="按阶段排列的知识模块"><p class="muted small-text">${mapMode==='knowledge'?'高亮模块共同涉及当前问题。点击模块，查找相关主题。':'点击模块，查看需要掌握的先修和后续应用。'}</p><div class="map-columns">${data.stages.map(s=>`<div class="map-column"><h3>${s.name}</h3>${data.modules.filter(x=>x.stage===s.id).map(x=>`<button class="map-node ${highlights.includes(x.id)?'highlight':''} ${mapMode==='prerequisite'&&x.id===m.id?'selected':''}" data-map-node="${x.id}" aria-pressed="${mapMode==='prerequisite'?x.id===m.id:highlights.includes(x.id)}"><span class="mono">${x.id}</span>${escape(x.title)}</button>`).join('')}</div>`).join('')}</div></section><section class="map-detail" aria-live="polite">${mapMode==='knowledge'?`<span class="tag blue">知识连接 · ${data.connections.indexOf(selected)+1} / ${data.connections.length}</span><h2>${selected.title}</h2><p>${selected.text}</p><div class="related-modules">${selected.modules.map(id=>`<a class="tag blue" href="${moduleLink(id)}">${id} ↗</a>`).join('')}</div><div class="notice neutral">${selected.evidence}</div><a class="button" href="${readLink(selected.source)}">沿证据继续阅读 →</a>${selected.lab?'<p style="margin-top:18px"><a href="#/lab">动手比较两种 loss 归一化 ↗</a></p>':''}`:`<span class="tag blue">${m.id} · 课程先修</span><h2>${escape(m.title)}</h2><h3>先掌握</h3><div class="related-modules">${upstream.length?upstream.map(id=>`<a class="tag blue" href="${moduleLink(id)}">${id} ${escape(moduleById(id).title)}</a>`).join(''):'<span class="muted">Python 基础；从入门诊断开始</span>'}</div><h3>后续应用</h3><div class="related-modules">${downstream.length?downstream.map(id=>`<a class="tag blue" href="${moduleLink(id)}">${id} ${escape(moduleById(id).title)}</a>`).join(''):'<span class="muted">综合项目与前沿专题</span>'}</div><p class="small-text">先修来自完整学习路线。M14 的基础评估从 M03 起使用，在后续阶段持续回访。</p><a class="button primary" href="${moduleLink(m.id)}">进入模块 →</a>`}</section></div>
  ${mapMode==='knowledge'?`<div class="section-title"><h2>沿一个问题，读懂多个系统</h2></div><div class="connection-list">${data.connections.map((c,i)=>`<button class="connection-item ${c.id===connectionId?'active':''}" data-connection="${c.id}" aria-pressed="${c.id===connectionId}"><span>0${i+1}</span>${c.title}</button>`).join('')}</div>`:''}<div class="notice neutral">知识联系、课程先修与软件依赖分别表示不同关系。高亮两个模块，不代表对应仓库可以直接安装在一起。</div>`;
}

function renderResources(){
  const libraries=[['handbook/00-end-to-end.md','超大 LLM 训练：端到端总览'],['handbook/01-data-model-pretraining-design.md','数据、模型与预训练设计'],['handbook/02-distributed-pretraining-operations.md','分布式预训练与运行'],['handbook/03-posttraining-agent-rl-evaluation.md','后训练、Agent RL 与评估'],['handbook/04-marin-535b-live-case-study.md','Marin 535B：公开训练案例'],['lessons/01-one-token-to-update.md','首课：一枚 token 到一次更新'],['RESOURCE_ATLAS.md','扩展资源：课程、论文与工具'],['GLOSSARY.md','训练与系统共用术语'],['KNOWLEDGE_TREE.md','完整知识树'],['COVERAGE.md','知识覆盖、深度与缺口']];
  const resources=data.resources.filter(r=>resourceFilter==='all'||r.group===resourceFilter).sort((a,b)=>a.firstModule.localeCompare(b.firstModule)||a.name.localeCompare(b.name));
  main.innerHTML=heading('Resource library / 一手资料','带着问题，进入源码。','每份资源都有建议入口、阅读问题和停止条件。先读透一个实现，再比较另一种设计。')+
  `<div class="section-title"><h2>连续阅读的手册与教材</h2><span class="tag">站内阅读全文</span></div><div class="reading-grid">${libraries.map(([file,title])=>`<a class="reading-link" href="${readLink(file)}">${title}<span aria-hidden="true">→</span></a>`).join('')}</div><div class="section-title"><h2>19 个固定源码项目</h2><span class="muted small-text">按首次学习模块排列</span></div><div class="filter-bar"><button class="chip ${resourceFilter==='all'?'active':''}" data-resource-filter="all" aria-pressed="${resourceFilter==='all'}">全部</button>${Object.entries(categoryLabels).map(([id,label])=>`<button class="chip ${resourceFilter===id?'active':''}" data-resource-filter="${id}" aria-pressed="${resourceFilter===id}">${label}</button>`).join('')}</div><div class="resource-grid">${resources.map(r=>`<article class="resource-card"><div class="resource-top"><span class="tag">${categoryLabels[r.group]}</span><a href="${moduleLink(r.firstModule)}">${r.firstModule} ↗</a></div><h3>${escape(r.name)}</h3><p>${escape(r.question)}</p><div class="stop"><span class="muted">第一轮读到：</span>${escape(r.stop)}</div><div class="resource-links"><a href="${readLink(r.note)}">阅读源码笔记 →</a><a href="${r.url}/tree/${r.commit}" target="_blank" rel="noopener noreferrer">固定源码 ↗</a></div><small class="source-version">${r.commit.slice(0,10)} · 局部源码审读</small></article>`).join('')}</div><div class="notice neutral">固定源码表示有版本记录和局部审读，完整运行与兼容性需要分别验证。<a href="${readLink('RESOURCE_ATLAS.md')}">查看更多外部一手资源及公开范围</a>。</div>`;
}

const quiz=[
  {q:'三个 logits 都为 0，正确标签为 B，交叉熵是多少？',options:['0','ln(3) ≈ 1.0986','3'],answer:1,why:'三个类别的概率均为 1/3，因此 loss = −ln(1/3) = ln(3)。'},
  {q:'logits 为 [2, 1, 0]，正确标签为 A，梯度的符号是什么？',options:['[−, +, +]','[+, −, +]','[+, +, −]'],answer:0,why:'梯度等于 p − onehot(A)。正确类别的梯度为负，梯度下降会提高它的 logit。'},
  {q:'两组有 1 和 3 个有效目标，局部平均 loss 为 0.4 和 2.4，全局 token 平均是多少？',options:['1.4','1.9','2.8'],answer:1,why:'(1 × 0.4 + 3 × 2.4) / 4 = 1.9。直接平均两组得到 1.4，表达的是另一个加权目标。'},
  {q:'一个 token 的 loss mask 为 0，是否意味着改变它作为输入上下文也不会影响模型？',options:['是，它完全被模型忽略','否，它仍可能影响后续预测'],answer:1,why:'忽略自身目标 loss 不等于屏蔽 attention；输入上下文仍可能改变后续预测。'}
];
function renderLab(){
  main.innerHTML=heading('Interactive lab / 动手理解','一枚 token，怎样改变参数？','调整 logits，观察概率、loss 与梯度，再比较不同分组下的训练目标。',`<a class="button" href="${readLink('lessons/01-one-token-to-update.md')}">阅读完整首课 →</a>`)+
  `<div class="lab-grid"><section class="panel"><span class="tag blue">01 · 输入</span><h2 style="margin-top:14px">三个候选 token</h2><p class="intro">Logit 是模型给每个候选的原始分数。</p>${['A','B','C'].map((name,i)=>`<div class="logit-control"><label for="logit-${i}">${name}</label><input type="range" id="logit-range-${i}" data-logit-range="${i}" min="-10" max="10" step="0.1" value="${Math.max(-10,Math.min(10,logits[i]))}" aria-label="Token ${name} logit 滑杆"><input id="logit-${i}" data-logit="${i}" type="number" min="-10000" max="10000" step="0.1" value="${logits[i]}" aria-label="Token ${name} logit 数值"></div>`).join('')}<label class="field"><span>正确标签</span><select id="target">${['A','B','C'].map((name,i)=>`<option value="${i}" ${target===i?'selected':''}>${name}</option>`).join('')}</select></label><label class="field"><span>学习率 η</span><input id="learning-rate" type="number" min="0" max="10" step="0.01" value="${rate}"></label><div class="lab-actions"><button class="button primary" data-action="step">执行一步更新 →</button><button class="button" data-action="shift">全部加 1000</button><button class="button ghost" data-action="reset-lab">恢复默认</button></div><p class="muted small-text" style="margin:16px 0 0">数值框支持更大范围；“全部加 1000”可检验概率对整体平移的不变性。</p><p id="lab-error" class="error small-text" role="status"></p></section><section class="panel" aria-label="计算结果"><span class="tag blue">02 · 从概率到梯度</span><div id="token-results" aria-live="polite"></div></section></div>
  <div class="notice neutral">教学模型直接把三个 logits 当作参数，使用解析梯度。它验证局部数学关系，没有实现 Transformer、autograd 或真实分布式训练。</div>
  <section class="panel"><span class="tag blue">03 · 一个贯穿数据与分布式的问题</span><h2 style="margin-top:14px">分组变了，目标也会变吗？</h2><p class="intro">六个位置共享上面的 logits。标签依次为 A、C、C、C、B、A；勾选参与 loss 的目标。</p><div class="mask-list">${['A','C','C','C','B','A'].map((label,i)=>`<label><input type="checkbox" data-mask="${i}" ${masks[i]?'checked':''}>${i+1} · ${label}</label>`).join('')}</div><div class="batch-controls"><label for="partition">分组方式</label><select id="partition"><option value="uneven" ${partition==='uneven'?'selected':''}>位置 1 / 位置 2–6（默认有效数 1:3）</option><option value="balanced" ${partition==='balanced'?'selected':''}>位置 1–2 / 位置 3–6（默认有效数 2:2）</option><option value="empty" ${partition==='empty'?'selected':''}>位置 1–4 / 位置 5–6（默认有效数 4:0）</option></select></div><div id="batch-results" aria-live="polite"></div><p class="muted small-text">这里约定目标是“全局有效 token 平均”。有意让各组等权也可以，但它定义了另一个目标。mask 为 0 仅表示不计该目标的 loss。</p></section>
  <div class="section-title"><h2>先预测，再核对</h2><span class="tag">4 道自测 · 可重试</span></div><form id="quiz-form" class="panel"><div class="quiz-list">${quiz.map((q,i)=>`<div class="quiz-item"><fieldset><legend>${i+1}. ${q.q}</legend><div class="quiz-options">${q.options.map((option,j)=>`<label><input type="radio" name="quiz-${i}" value="${j}" required>${option}</label>`).join('')}</div><p class="quiz-feedback" id="quiz-feedback-${i}" hidden></p></fieldset></div>`).join('')}</div><button class="button primary" type="submit">核对答案</button><p id="quiz-score" class="small-text" role="status" style="margin:15px 0 0"></p><p class="muted small-text" style="margin:12px 0 0">自测帮助检查理解，不会自动标记模块结业。解释机制、完成练习并保留证据后，再自主验收。</p></form>`;
  updateLab();
}
function fixed(value,digits=6){return Number(value).toFixed(digits);}
function updateLab(){
  const error=document.getElementById('lab-error');if(!error)return;
  try{
    const result=tokenStep(logits,target,rate),after={loss:result.nextLoss};
    error.textContent='';
    document.getElementById('token-results').innerHTML=`<div class="probabilities">${result.probabilities.map((p,i)=>`<div class="prob-row ${i===target?'target':''}"><span class="prob-label">${['A','B','C'][i]}${i===target?' ✓':''}</span><div class="prob-track" role="meter" aria-label="Token ${['A','B','C'][i]} 概率" aria-valuemin="0" aria-valuemax="1" aria-valuenow="${p}"><div class="prob-bar" style="width:${p*100}%"></div></div><span class="prob-value">${(p*100).toFixed(2)}%</span></div>`).join('')}</div><div class="metric-row"><div><small>当前交叉熵 loss</small><strong>${fixed(result.loss)}</strong></div><div><small>下一步预期 loss</small><strong>${fixed(after.loss)}</strong></div></div><div class="math-line">gradient = p − onehot(y)</div><div class="table-scroll"><table class="data-table"><thead><tr><th>Token</th><th>梯度</th><th>更新后 logit</th></tr></thead><tbody>${result.gradient.map((g,i)=>`<tr><td>${['A','B','C'][i]}</td><td class="mono">${g>=0?'+':''}${fixed(g)}</td><td class="mono">${fixed(result.next[i])}</td></tr>`).join('')}</tbody></table></div><p class="muted small-text">z′ = z − η × gradient。正梯度让对应 logit 降低，负梯度让它提高。</p>${lastStep?`<div class="notice">刚才一步：loss ${fixed(lastStep.before)} → ${fixed(lastStep.after)}。上方显示更新后的当前状态。</div>`:''}`;
    const batch=batchComparison(logits,masks,partition);
    document.getElementById('batch-results').innerHTML=!batch.count?'<div class="notice warning">没有有效目标，无法定义平均 loss。请至少勾选一个目标；不能把空批次的 loss 记作 0。</div>':`<p class="small-text">有效目标：<strong>${batch.count}</strong> 个 · 两组分别为 <strong>${batch.groups.map(g=>g.count).join(' : ')}</strong>。${batch.groups.some(g=>!g.count)?'空组不参与平均，贡献为 0。':''}</p><div class="table-scroll"><table class="data-table"><thead><tr><th>聚合规则</th><th>Loss</th><th>梯度 [A, B, C]</th></tr></thead><tbody>${[['全局 token 平均',batch.global],['按有效数量加权',batch.weighted],['局部均值直接平均',batch.equalGroups]].map(([label,value])=>`<tr><td>${label}</td><td class="mono">${fixed(value.loss)}</td><td class="mono">[${value.gradient.map(g=>fixed(g,4)).join(', ')}]</td></tr>`).join('')}</tbody></table></div><div class="math-line">L = Σ (n_group × L_group) / Σ n_group</div><p class="${Math.abs(batch.global.loss-batch.equalGroups.loss)>1e-8?'error':'good'} small-text">${Math.abs(batch.global.loss-batch.equalGroups.loss)>1e-8?'本例中，直接平均局部均值改变了目标。':'本例两种平均的 loss 相同；是否具有同一目标，还要检查权重与梯度。'}</p>`;
  }catch(e){error.textContent=e.message;document.getElementById('token-results').innerHTML='<p class="muted">修正输入后继续计算。</p>';document.getElementById('batch-results').innerHTML='';}
}

function renderProgress(){
  const records=data.modules.filter(m=>entry(m.id).status!=='new'||entry(m.id).note||entry(m.id).evidence);
  const done=records.filter(m=>entry(m.id).status==='reviewed').length,learning=records.filter(m=>entry(m.id).status==='learning').length;
  main.innerHTML=heading('My learning / 个人记录','把理解，变成可检查的能力。','记录机制解释、代码实现、实验结果与未解决问题。进度由你自主验收。')+
  `<div class="progress-overview"><div class="progress-stat"><strong>${learning}</strong><span>学习中</span></div><div class="progress-stat"><strong>${done}<small> / 16</small></strong><span>已自主验收</span></div><div class="progress-stat"><strong>${records.filter(m=>entry(m.id).note||entry(m.id).evidence).length}</strong><span>有记录的模块</span></div></div><div class="notice ${storageAvailable?'neutral':'warning'}">${storageAvailable?'记录仅保存在当前浏览器，不会上传到 GitHub。清理浏览器数据或换设备时不会自动同步，请导出备份。':'当前浏览器无法持久保存，请及时导出记录。'}</div><div class="filter-bar"><button class="button primary" data-action="export-progress">导出学习记录 ↓</button><button class="button" data-action="import-progress">导入并合并记录 ↑</button><input type="file" id="progress-file" accept=".json,application/json" hidden><a class="button ghost" href="${readLink('templates/learner-progress.md')}">Markdown 记录模板 ↗</a></div><p class="muted small-text">导入会保留其他模块，并更新文件中同名模块的记录；仅导入你信任的个人备份。</p>
  ${records.length?records.map(m=>{const r=entry(m.id);return `<article class="progress-entry"><header><h3><a href="${moduleLink(m.id)}"><span class="mono">${m.id}</span> ${escape(m.title)} →</a></h3>${statusTag(m.id)}</header>${r.note?`<p class="saved-note">${escape(r.note)}</p>`:''}${r.evidence?`<details><summary>验收证据</summary><p class="saved-note">${escape(r.evidence)}</p></details>`:''}${r.updated?`<time datetime="${escape(r.updated)}">更新：${escape(new Date(r.updated).toLocaleString('zh-CN'))}</time>`:''}</article>`;}).join(''):'<div class="empty-state"><h2>从一个具体问题开始。</h2><p>进入模块，写下第一条理解或疑问。记录会出现在这里。</p><a class="button primary" href="#/module/M00">开始 M00 →</a></div>'}`;
}
function renderSearch(query){
  const results=searchDocuments(data.docs,query);
  main.innerHTML=heading('Search / 全文检索',query?'搜索「'+query+'」':'搜索学习资料',query?`找到 ${results.length} 篇相关资料。支持中文、英文、模块编号与代码标识。`:'在顶部输入想理解的概念，例如 loss mask、checkpoint 或 GRPO。')+
  (results.length?results.map(r=>`<a class="search-result" href="${readLink(r.path)}"><small>${escape(r.path)}</small><h2>${escape(r.title)}</h2><p>${escape(r.snippet)}…</p></a>`).join(''):`<div class="empty-state"><p>${query?'暂时没有匹配内容。试试更短的关键词或英文术语。':'从一个具体概念开始搜索。'}</p><a href="#/resources">浏览资源与手册 →</a></div>`);
}
function renderMissing(){main.innerHTML=heading('Page not found','这份内容暂时没有找到。','链接可能已更新。可以回到学习路径，或搜索文档标题。', '<a class="button primary" href="#/learn">返回学习路径 →</a>');}

async function navigate(){
  if(!data)return;
  const version=++routeVersion;
  const route=location.hash.slice(2)||'learn';const [pathname,query='']=route.split('?');const [name,...parts]=pathname.split('/');const params=new URLSearchParams(query);
  let file;try{file=decodeURIComponent(parts.join('/'));}catch{return renderMissing();}
  document.getElementById('current-section').textContent=titleNames[name]||'学习手册';
  document.title=(name==='module'?moduleById(file)?.title:name==='read'?(Object.hasOwn(data.docs,file)?data.docs[file].title:'未找到内容'):titleNames[name])+' · Frontier LLM Handbook';
  document.querySelectorAll('[data-nav]').forEach(a=>{const active=a.dataset.nav===(name==='module'?'learn':name==='read'?'resources':name);a.classList.toggle('active',active);if(active)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});
  document.getElementById('sidebar').classList.remove('open');document.getElementById('menu-toggle').setAttribute('aria-expanded','false');
  if(name==='learn')renderLearn();else if(name==='module')renderModule(file);else if(name==='read')renderDocument(file,params.get('anchor'));else if(name==='map')renderMap();else if(name==='resources')renderResources();else if(name==='lab')renderLab();else if(name==='progress')renderProgress();else if(name==='search')renderSearch(params.get('q')||'');else renderMissing();
  updateSidebar();window.scrollTo(0,0);main.focus({preventScroll:true});
  await renderMermaid(version);
  if(version===routeVersion&&name==='read'&&params.get('anchor'))scrollHeading(params.get('anchor'));
}

function saveModule(id,explicit=true){
  const status=document.getElementById('module-status').value,note=document.getElementById('module-note').value,evidence=document.getElementById('module-evidence').value;
  if(status==='reviewed'&&!evidence.trim()){toast('请先填写验收证据，再标记为已自主验收。');document.getElementById('module-evidence').focus();return;}
  progress.modules[id]={status,note,evidence,updated:new Date().toISOString()};
  const saved=saveProgress();updateSidebar();
  document.getElementById('save-feedback').textContent=saved?'已保存到当前浏览器':'暂存于本次页面，请导出备份';
  if(explicit)toast(saved?'学习记录已保存。':'请在“我的学习”导出本次记录。');
}
function syncLogitInputs(){logits.forEach((value,i)=>{document.getElementById('logit-'+i).value=String(value);const range=document.getElementById('logit-range-'+i);range.value=String(Math.max(-10,Math.min(10,value)));range.disabled=Math.abs(value)>10;});}

main.addEventListener('click',async event=>{
  const el=event.target.closest('button');if(!el)return;
  if(el.dataset.stage){stageFilter=el.dataset.stage;renderLearn();return;}
  if(el.dataset.resourceFilter){resourceFilter=el.dataset.resourceFilter;renderResources();return;}
  if(el.dataset.mapMode){mapMode=el.dataset.mapMode;renderMap();return;}
  if(el.dataset.connection){connectionId=el.dataset.connection;renderMap();return;}
  if(el.dataset.mapNode){selectedModule=el.dataset.mapNode;if(mapMode==='knowledge'){const c=data.connections.find(c=>c.modules.includes(selectedModule));if(c)connectionId=c.id;else{mapMode='prerequisite';}}renderMap();return;}
  if(el.dataset.heading){scrollHeading(el.dataset.heading);return;}
  const action=el.dataset.action;
  if(action==='save-module')saveModule(el.dataset.id);
  if(action==='step'){
    try{const before=tokenStep(logits,target,rate);if(before.next.some(v=>Math.abs(v)>10000)){toast('下一步超出输入范围，请先调整 logits 或降低学习率。');return;}logits=before.next;lastStep={before:before.loss,after:before.nextLoss};syncLogitInputs();updateLab();}catch(e){toast(e.message);}
  }
  if(action==='shift'){
    if(logits.some(v=>!Number.isFinite(v)||Math.abs(v+1000)>10000)){toast('平移后超出教学范围，请先调整 logits。');return;}
    logits=logits.map(v=>v+1000);lastStep=null;syncLogitInputs();updateLab();
  }
  if(action==='reset-lab'){logits=[2,1,0];target=1;rate=.1;masks=[1,1,1,1,0,0];partition='uneven';lastStep=null;renderLab();}
  if(action==='export-progress'){
    const blob=new Blob([JSON.stringify(progress,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download='frontier-llm-learning-'+new Date().toISOString().slice(0,10)+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  if(action==='import-progress')document.getElementById('progress-file').click();
});

main.addEventListener('input',event=>{
  const el=event.target;
  if(el.dataset.logit!==undefined){logits[Number(el.dataset.logit)]=el.value.trim()===''?NaN:Number(el.value);lastStep=null;const range=document.getElementById('logit-range-'+el.dataset.logit);range.value=String(Math.max(-10,Math.min(10,logits[Number(el.dataset.logit)])));range.disabled=Math.abs(logits[Number(el.dataset.logit)])>10;updateLab();}
  if(el.dataset.logitRange!==undefined){logits[Number(el.dataset.logitRange)]=Number(el.value);document.getElementById('logit-'+el.dataset.logitRange).value=el.value;lastStep=null;updateLab();}
  if(el.id==='learning-rate'){rate=el.value.trim()===''?NaN:Number(el.value);lastStep=null;updateLab();}
  if(el.id==='module-note'||el.id==='module-evidence'){
    const id=el.dataset.module,old=entry(id);
    progress.modules[id]={...old,[el.id==='module-note'?'note':'evidence']:el.value,updated:new Date().toISOString()};
    if(progress.modules[id].status==='reviewed'&&!progress.modules[id].evidence.trim()){progress.modules[id].status='learning';document.getElementById('module-status').value='learning';}
    const saved=saveProgress();document.getElementById('save-feedback').textContent=saved?'笔记已自动保存；状态修改请点保存':'暂存于本次页面，请导出备份';updateSidebar();
  }
});
main.addEventListener('change',async event=>{
  const el=event.target;
  if(el.id==='target'){target=Number(el.value);lastStep=null;updateLab();}
  if(el.dataset.mask!==undefined){masks[Number(el.dataset.mask)]=Number(el.checked);updateLab();}
  if(el.id==='partition'){partition=el.value;updateLab();}
  if(el.id==='progress-file'&&el.files?.[0]){
    try{if(el.files[0].size>1000000)throw new Error('备份文件过大，请选择本站导出的学习记录。');const incoming=validateProgress(JSON.parse(await el.files[0].text()));progress={...progress,modules:{...progress.modules,...incoming.modules},lastModule:incoming.lastModule||progress.lastModule};const saved=saveProgress();renderProgress();updateSidebar();toast(saved?'学习记录已导入并合并。':'记录已导入本次页面，请及时导出备份。');}catch(e){toast('导入失败：'+e.message);}
  }
});
main.addEventListener('submit',event=>{
  if(event.target.id!=='quiz-form')return;event.preventDefault();const values=new FormData(event.target);let correct=0;
  quiz.forEach((q,i)=>{const ok=Number(values.get('quiz-'+i))===q.answer;correct+=Number(ok);const output=document.getElementById('quiz-feedback-'+i);output.hidden=false;output.className='quiz-feedback '+(ok?'good':'error');output.textContent=(ok?'✓ 正确。':'再想一想。')+q.why;});
  document.getElementById('quiz-score').textContent=`本次答对 ${correct} / 4。${correct===4?'接下来试着不看公式解释机制，并完成模块练习。':'对照解释修改答案，可以再次核对。'}`;
});
document.getElementById('search-form').addEventListener('submit',event=>{event.preventDefault();location.hash='#/search?q='+encodeURIComponent(document.getElementById('global-search').value.trim());});
document.getElementById('menu-toggle').addEventListener('click',()=>{const open=document.getElementById('sidebar').classList.toggle('open');document.getElementById('menu-toggle').setAttribute('aria-expanded',String(open));});
document.addEventListener('keydown',event=>{
  if(event.key==='/'&&!['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)){event.preventDefault();document.getElementById('global-search').focus();}
  if(event.key==='Escape'){document.getElementById('sidebar').classList.remove('open');document.getElementById('menu-toggle').setAttribute('aria-expanded','false');}
});
window.addEventListener('hashchange',navigate);
window.addEventListener('storage',event=>{
  if(event.key!==storageKey)return;
  try{
    const id=document.getElementById('module-note')?.dataset.module;
    const previous=id?JSON.stringify(entry(id)):null;
    progress=event.newValue?validateProgress(JSON.parse(event.newValue)):emptyProgress();
    updateSidebar();
    if(id&&previous!==JSON.stringify(entry(id))){
      const updated=entry(id);
      document.getElementById('module-note').value=updated.note;
      document.getElementById('module-evidence').value=updated.evidence;
      document.getElementById('module-status').value=updated.status;
      document.getElementById('save-feedback').textContent='已同步另一页面的最新记录';
      toast('另一页面更新了本模块，已同步最新记录。');
    }
    if(location.hash==='#/progress')renderProgress();
  }catch{toast('另一页面的记录格式无法识别，请先导出当前记录。');}
});

try{const saved=localStorage.getItem(storageKey);if(saved)progress=validateProgress(JSON.parse(saved));}catch{storageAvailable=false;toast('未能读取浏览器中的学习记录，请使用备份导入。');}
try{
  const response=await fetch(new URL('./content.json',document.baseURI));if(!response.ok)throw new Error('资料请求失败 '+response.status);data=await response.json();
  if(data.schemaVersion!==1||data.modules?.length!==16)throw new Error('资料格式不兼容');
  await navigate();
}catch(error){main.innerHTML=heading('Loading error','学习资料暂时无法加载。','请刷新页面重试，或先在 GitHub 阅读原文。',`<a class="button primary" href="https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/ROADMAP.md">阅读完整学习路线 ↗</a>`);console.error(error);}

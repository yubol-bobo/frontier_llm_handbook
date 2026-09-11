import {t,ui,getLocale,setLocale,chooseLocale,languageKey} from './i18n.mjs';
import {initializeTheme,observeReading,mountMapWires} from './interactions.mjs';
import {tokenResultsShell,updateTokenResults,batchResultsHtml} from './lab-view.mjs';
import {icon,mountIcons,renderLearningView,focusPreservingRender} from './design.mjs';
import {tokenStep,batchComparison,statuses,emptyProgress,validateProgress,searchDocuments} from './learning.mjs';

let savedLanguage;try{savedLanguage=localStorage.getItem(languageKey);}catch{}
setLocale(chooseLocale(location.href,savedLanguage));
const main=document.getElementById('main-content');
const storageKey='frontier-llm-handbook:learning:v1';
let data,progress=emptyProgress(),storageAvailable=true,routeVersion=0,toastTimer;
let stageFilter='all',resourceFilter='all',mapMode='knowledge',connectionId='loss',selectedModule='M02';
let logits=[2,1,0],target=1,rate=0.1,masks=[1,1,1,1,0,0],partition='uneven',lastStep=null;
let lossHistory=[],readerCleanup=()=>{},mapController=null;
let diagramVersion=0,diagramQueue=Promise.resolve();
const contentCache=new Map();
const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const readLink=(file,anchor='')=>'#/read/'+file.split('/').map(encodeURIComponent).join('/')+(anchor?'?anchor='+encodeURIComponent(anchor):'');
const moduleLink=id=>'#/module/'+id;
const entry=id=>progress.modules[id]||{status:'new',note:'',evidence:'',updated:''};
const moduleById=id=>data.modules.find(m=>m.id===id);
const titleNames=()=>({learn:t('学习路径'),module:t('课程阅读'),read:t('资料阅读'),map:t('知识地图'),resources:t('资源与手册'),lab:t('互动实验'),progress:t('我的学习'),search:t('搜索')});
const categoryLabels=()=>({training:t('模型与训练'),infra:t('基础设施'),harness:'Agent Harness',rl:t('强化学习与环境')});

function toast(message){const element=document.getElementById('toast');element.textContent=message;element.hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>element.hidden=true,4200);}
function saveProgress(){
  try{localStorage.setItem(storageKey,JSON.stringify(progress));storageAvailable=true;return true;}
  catch{storageAvailable=false;toast(t('浏览器未能保存记录；本次仍可学习，请及时导出。'));return false;}
}
function updateSidebar(){
  document.querySelectorAll('[data-status-module]').forEach(element=>{
    const status=entry(element.dataset.statusModule).status;
    element.textContent=t(statuses[status]);
    element.className='tag '+(status==='reviewed'?'green':status==='learning'?'blue':'');
  });
  const done=data.modules.filter(m=>entry(m.id).status==='reviewed').length;
  document.getElementById('side-progress').innerHTML=ui`<div class="progress-meta"><span>我的核心课程</span><span class="mono">${done} / 16</span></div><div class="progress-track" role="progressbar" aria-label="自主验收进度" aria-valuemin="0" aria-valuemax="16" aria-valuenow="${done}"><div class="progress-fill" style="width:${done/16*100}%"></div></div><small>自主验收 · 仅保存在当前浏览器</small>`;
}
function heading(eyebrow,title,text='',action=''){
  return `<div class="page-heading"><div><div class="eyebrow">${escape(eyebrow)}</div><h1>${escape(title)}</h1>${text?`<p>${escape(text)}</p>`:''}</div>${action}</div>`;
}
function statusTag(id){const status=entry(id).status;return `<span data-status-module="${id}" class="tag ${status==='reviewed'?'green':status==='learning'?'blue':''}">${t(statuses[status])}</span>`;}

function renderLearn(){main.innerHTML=renderLearningView({data,progress,entry,readLink,moduleLink,statusTag,stageFilter});}

function notePanel(id){const saved=entry(id);return ui`<details class="note-panel" open><summary>我的模块记录</summary><small>笔记仅存在当前浏览器。可在“我的学习”导出备份。</small><label for="module-status">学习状态</label><select id="module-status" data-module="${id}">${Object.entries(statuses).map(([value,label])=>`<option value="${value}" ${saved.status===value?'selected':''}>${t(label)}</option>`).join('')}</select><label for="module-note">理解、疑问与工程记录</label><textarea id="module-note" data-module="${id}" maxlength="20000" placeholder="我能解释什么？实现或排障时发现了什么？">${escape(saved.note)}</textarea><label for="module-evidence">验收证据</label><textarea class="evidence" id="module-evidence" data-module="${id}" maxlength="5000" placeholder="代码或笔记链接、验证结果与未完成部分">${escape(saved.evidence)}</textarea><small>“已自主验收”需填写证据，并自行核对模块出口要求。</small><button class="button small" data-action="save-module" data-id="${id}">保存状态与记录</button><div id="save-feedback" class="save-feedback" role="status"></div></details>`;}
function toc(headings){return ui`<div><h3>本页目录</h3><nav class="toc" aria-label="本页目录">${headings.filter(h=>h.level<=3).map(h=>`<button data-heading="${escape(h.id)}">${escape(h.label)}</button>`).join('')}</nav></div>`;}
function readerTools(){
  main.querySelectorAll('.markdown pre:not(.mermaid)').forEach(pre=>{
    const toolbar=document.createElement('div');toolbar.className='code-toolbar';
    const button=document.createElement('button');button.type='button';button.textContent=t('复制');button.setAttribute('aria-label',t('复制代码块'));
    button.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(pre.textContent);button.textContent=t('已复制');setTimeout(()=>button.textContent=t('复制'),1800);}catch{toast(t('无法自动复制，请选中代码复制。'));}});
    toolbar.append(button);pre.before(toolbar);
  });
}
function renderMermaid(version){
  const ticket=++diagramVersion;
  const render=async()=>{
  const nodes=[...main.querySelectorAll('pre.mermaid,div.mermaid[data-source]')];if(!nodes.length)return;
  const obsolete=()=>version!==routeVersion||ticket!==diagramVersion;
  try{
    const {default:mermaid}=await import('mermaid');
    if(obsolete())return;
    const dark=document.documentElement.dataset.theme==='dark';
    const palette=dark?{darkMode:true,background:'#151821',primaryColor:'#262338',primaryTextColor:'#eeeef6',primaryBorderColor:'#9180cb',secondaryColor:'#1d302c',tertiaryColor:'#242b40',lineColor:'#a0a7bb',textColor:'#eeeef6',clusterBkg:'#1c202c',clusterBorder:'#3c4358',edgeLabelBackground:'#151821',titleColor:'#eeeef6'}:{darkMode:false,background:'#fff',primaryColor:'#f0ebfc',primaryTextColor:'#232636',primaryBorderColor:'#a493ce',secondaryColor:'#eaf5f0',tertiaryColor:'#eef1fa',lineColor:'#737c91',textColor:'#232636',clusterBkg:'#f2f4f9',clusterBorder:'#c4cbdc',edgeLabelBackground:'#fff',titleColor:'#232636'};
    mermaid.initialize({startOnLoad:false,securityLevel:'strict',theme:'base',themeVariables:{...palette,fontFamily:'Geist Variable, Noto Sans SC Variable, sans-serif',fontSize:'16px'},flowchart:{htmlLabels:false,useMaxWidth:true},suppressErrorRendering:true});
    for(let i=0;i<nodes.length;i++){
      if(obsolete())return;
      const source=nodes[i].dataset.source||nodes[i].textContent;
      try{
        const {svg}=await mermaid.render('diagram-'+ticket+'-'+i,source);
        if(obsolete())return;
        if(nodes[i].tagName==='DIV'){nodes[i].innerHTML=svg;continue;}
        const diagram=document.createElement('div');diagram.className='mermaid';diagram.dataset.source=source;diagram.innerHTML=svg;diagram.setAttribute('role','img');diagram.setAttribute('aria-label',t('文档关系图，展开下方文字可阅读图的原始关系'));
        const details=document.createElement('details');const summary=document.createElement('summary');summary.textContent=t('查看图的文字关系');const pre=document.createElement('pre');pre.textContent=source;details.dataset.diagramIndex=String(i);details.append(summary,pre);nodes[i].replaceWith(diagram,details);
      }catch{if(obsolete())return;if(nodes[i].tagName==='PRE'){nodes[i].classList.remove('mermaid');nodes[i].setAttribute('aria-label',t('关系图文字源，图形暂无法显示'));}}
    }
  }catch{if(!obsolete())toast(t('关系图暂时无法加载，已保留文字关系。'));}
  };
  diagramQueue=diagramQueue.then(render,render);return diagramQueue;
}
function renderModule(id){
  const m=moduleById(id);if(!m)return renderMissing();
  progress.lastModule=id;saveProgress();
  const index=data.modules.indexOf(m),stage=data.stages.find(s=>s.id===m.stage);
  main.innerHTML=ui`<div class="reader-meta"><a href="#/learn">← 学习路径</a><span class="tag">${stage.name}</span><span class="tag blue">${id}</span>${statusTag(id)}</div><div class="reader-heading"><h1>${escape(m.title)}</h1><p>${escape(m.question)}</p><div class="prerequisites"><span class="muted">先修：</span>${m.prerequisites.length?m.prerequisites.map(p=>`<a href="${moduleLink(p)}">${p} ${escape(moduleById(p).title)}</a>`).join(''):t('Python 基础；按模块诊断补齐')}<a class="source-link" href="${data.repoUrl}/blob/main/${m.file}#${m.anchor}" target="_blank" rel="noopener noreferrer">原始课程 ↗</a></div></div>
  <div class="reader-layout"><div class="reader-body"><div class="notice neutral">这是课程要求与练习设计；完成阅读后，请通过实现、解释和证据验收自己的能力。<a href="${readLink('COVERAGE.md')}">查看教材与实验完成范围</a>。</div><article class="markdown">${m.html.replace(/<h2[^>]*>.*?<\/h2>\s*/s,'')}</article><nav class="reading-actions" aria-label="模块前后导航">${index?ui`<a class="button" href="${moduleLink(data.modules[index-1].id)}">← ${data.modules[index-1].id} 上一模块</a>`:t('<a class="button" href="#/lab">体验互动实验</a>')}${index<15?ui`<a class="button primary" href="${moduleLink(data.modules[index+1].id)}">${data.modules[index+1].id} 下一模块 →</a>`:t('<a class="button primary" href="#/progress">整理综合项目 →</a>')}</nav></div><aside class="reader-aside">${toc(m.headings.slice(1))}${notePanel(id)}</aside></div>`;
  readerTools();
}
function renderDocument(file,anchor){
  const doc=Object.hasOwn(data.docs,file)?data.docs[file]:null;if(!doc)return renderMissing();
  main.innerHTML=ui`<div class="reader-meta"><a href="#/resources">← 资源与手册</a><span class="tag">约 ${doc.minutes} 分钟阅读 · 不含练习</span><a class="source-link" href="${data.repoUrl}/blob/main/${encodeURI(file)}${anchor?'#'+encodeURIComponent(anchor):''}" target="_blank" rel="noopener noreferrer">在 GitHub 查看原文 ↗</a></div><div class="reader-layout"><div class="reader-body"><article class="markdown">${doc.html}</article></div><aside class="reader-aside">${toc(doc.headings)}<div class="notice neutral">内容来自资料库原文。<a href="#/learn">回到学习路径</a>，可按模块保存笔记与验收记录。</div></aside></div>`;
  readerTools();
}
function scrollHeading(id){const element=document.getElementById(id);if(element&&main.contains(element))element.scrollIntoView({block:'start',behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});}

function renderMap(){
  const selected=data.connections.find(c=>c.id===connectionId)||data.connections[0];
  const m=moduleById(selectedModule)||data.modules[2];
  const upstream=m.prerequisites,downstream=data.prerequisites.filter(p=>p[0]===m.id).map(p=>p[1]);
  const highlights=mapMode==='knowledge'?selected.modules:[m.id,...upstream,...downstream];
  const markup=heading(t('Knowledge map / 跨层理解'),t('把知识连起来。'),t('选择一个问题，观察它怎样跨越课程阶段，再沿证据回到实际实现。'))+
  ui`<div class="filter-bar"><button class="chip ${mapMode==='knowledge'?'active':''}" data-map-mode="knowledge" aria-pressed="${mapMode==='knowledge'}">跨模块知识联系</button><button class="chip ${mapMode==='prerequisite'?'active':''}" data-map-mode="prerequisite" aria-pressed="${mapMode==='prerequisite'}">课程先修关系</button><a class="button ghost small" href="${readLink('REPO_RELATIONSHIPS.md')}">查看真实软件依赖 ↗</a></div><div class="map-layout"><section class="map-board" aria-label="按阶段排列的知识模块"><p class="muted small-text">${mapMode==='knowledge'?t('高亮模块共同涉及当前问题。点击模块，查找相关主题。'):t('点击模块，查看需要掌握的先修和后续应用。')}</p><svg class="map-wires" aria-hidden="true"></svg><div class="map-topic" ${mapMode==='knowledge'?'':'hidden'}>${icon('network')}<span>${escape(selected.title)}</span></div><div class="map-columns">${data.stages.map(s=>`<div class="map-column"><h3>${s.name}</h3>${data.modules.filter(x=>x.stage===s.id).map(x=>`<button class="map-node ${highlights.includes(x.id)?'highlight':'unrelated'} ${mapMode==='prerequisite'&&x.id===m.id?'selected':''}" data-map-node="${x.id}" aria-pressed="${mapMode==='prerequisite'?x.id===m.id:highlights.includes(x.id)}"><span class="mono">${x.id}</span>${escape(x.title)}</button>`).join('')}</div>`).join('')}</div></section><section class="map-detail" aria-live="polite">${mapMode==='knowledge'?ui`<span class="tag blue">知识连接 · ${data.connections.indexOf(selected)+1} / ${data.connections.length}</span><h2>${selected.title}</h2><p>${selected.text}</p><div class="related-modules">${selected.modules.map(id=>`<a class="tag blue" href="${moduleLink(id)}">${id} ↗</a>`).join('')}</div><div class="notice neutral">${selected.evidence}</div><a class="button" href="${readLink(selected.source)}">沿证据继续阅读 →</a>${selected.lab?t('<p style="margin-top:18px"><a href="#/lab">动手比较两种 loss 归一化 ↗</a></p>'):''}`:ui`<span class="tag blue">${m.id} · 课程先修</span><h2>${escape(m.title)}</h2><h3>先掌握</h3><div class="related-modules">${upstream.length?upstream.map(id=>`<a class="tag blue" href="${moduleLink(id)}">${id} ${escape(moduleById(id).title)}</a>`).join(''):t('<span class="muted">Python 基础；从入门诊断开始</span>')}</div><h3>后续应用</h3><div class="related-modules">${downstream.length?downstream.map(id=>`<a class="tag blue" href="${moduleLink(id)}">${id} ${escape(moduleById(id).title)}</a>`).join(''):t('<span class="muted">综合项目与前沿专题</span>')}</div><p class="small-text">先修来自完整学习路线。M14 的基础评估从 M03 起使用，在后续阶段持续回访。</p><a class="button primary" href="${moduleLink(m.id)}">进入模块 →</a>`}</section></div>
  ${mapMode==='knowledge'?ui`<div class="section-title"><h2>沿一个问题，读懂多个系统</h2></div><div class="connection-list">${data.connections.map((c,i)=>`<button class="connection-item ${c.id===connectionId?'active':''}" data-connection="${c.id}" aria-pressed="${c.id===connectionId}"><span>0${i+1}</span>${c.title}</button>`).join('')}</div>`:''}<div class="notice neutral">知识联系、课程先修与软件依赖分别表示不同关系。高亮两个模块，不代表对应仓库可以直接安装在一起。</div>`;
  const oldBoard=main.querySelector('.map-board');
  if(oldBoard){
    const draft=document.createElement('div');draft.innerHTML=markup;
    main.querySelector('.map-detail').innerHTML=draft.querySelector('.map-detail').innerHTML;
    main.querySelector('.filter-bar').replaceWith(draft.querySelector('.filter-bar'));
    oldBoard.querySelector('p').textContent=draft.querySelector('.map-board>p').textContent;
    const newTopic=draft.querySelector('.map-topic'),oldTopic=oldBoard.querySelector('.map-topic');
    oldTopic.innerHTML=newTopic.innerHTML;oldTopic.hidden=newTopic.hidden;
    oldBoard.querySelectorAll('[data-map-node]').forEach(node=>{
      const replacement=draft.querySelector(`[data-map-node="${node.dataset.mapNode}"]`);
      node.className=replacement.className;node.setAttribute('aria-pressed',replacement.getAttribute('aria-pressed'));
    });
    const oldConnections=main.querySelector('.connection-list');
    const newConnections=draft.querySelector('.connection-list');
    if(oldConnections){oldConnections.previousElementSibling.remove();oldConnections.remove();}
    if(newConnections){main.lastElementChild.before(newConnections.previousElementSibling,newConnections);}
    if(mapController)mapController.redraw();
    else mapController=mountMapWires(oldBoard,()=>({mode:mapMode,prerequisites:data.prerequisites,selected:data.connections.find(c=>c.id===connectionId),selectedModule}));
  }else{
    main.innerHTML=markup;
    mapController=mountMapWires(main.querySelector('.map-board'),()=>({mode:mapMode,prerequisites:data.prerequisites,selected:data.connections.find(c=>c.id===connectionId),selectedModule}));
  }

}

function renderResources(){
  const libraries=[['handbook/06-sol-pi-efficient-harnesses.md',t('SoL-Pi：可验证的 Harness 提效')],['handbook/05-claude-code-harness.md',t('Claude Code：执行循环、上下文与恢复')],['handbook/00-end-to-end.md',t('超大 LLM 训练：端到端总览')],['handbook/01-data-model-pretraining-design.md',t('数据、模型与预训练设计')],['handbook/02-distributed-pretraining-operations.md',t('分布式预训练与运行')],['handbook/03-posttraining-agent-rl-evaluation.md',t('后训练、Agent RL 与评估')],['handbook/04-marin-535b-live-case-study.md',t('Marin 535B：公开训练案例')],['lessons/01-one-token-to-update.md',t('首课：一枚 token 到一次更新')],['RESOURCE_ATLAS.md',t('扩展资源：课程、论文与工具')],['GLOSSARY.md',t('训练与系统共用术语')],['KNOWLEDGE_TREE.md',t('完整知识树')],['COVERAGE.md',t('知识覆盖、深度与缺口')]];
  const resources=data.resources.filter(r=>resourceFilter==='all'||r.group===resourceFilter).sort((a,b)=>a.firstModule.localeCompare(b.firstModule)||a.name.localeCompare(b.name));
  main.innerHTML=heading(t('Resource library / 一手资料'),t('带着问题，进入源码。'),t('每份资源都有建议入口、阅读问题和停止条件。先读透一个实现，再比较另一种设计。'))+
  ui`<div class="section-title"><h2>连续阅读的手册与教材</h2><span class="tag">站内阅读全文</span></div><div class="reading-grid">${libraries.map(([file,title])=>`<a class="reading-link" href="${readLink(file)}">${title}<span aria-hidden="true">→</span></a>`).join('')}</div><div class="section-title"><h2>${data.resources.length} 个固定源码项目</h2><span class="muted small-text">按首次学习模块排列</span></div><div class="filter-bar"><button class="chip ${resourceFilter==='all'?'active':''}" data-resource-filter="all" aria-pressed="${resourceFilter==='all'}">全部</button>${Object.entries(categoryLabels()).map(([id,label])=>`<button class="chip ${resourceFilter===id?'active':''}" data-resource-filter="${id}" aria-pressed="${resourceFilter===id}">${label}</button>`).join('')}</div><div class="resource-grid">${resources.map(r=>ui`<article class="resource-card"><div class="resource-top"><span class="tag">${categoryLabels()[r.group]}</span><a href="${moduleLink(r.firstModule)}">${r.firstModule} ↗</a></div><h3>${escape(r.name)}</h3><p>${escape(r.question)}</p><div class="stop"><span class="muted">第一轮读到：</span>${escape(r.stop)}</div><div class="resource-links"><a href="${readLink(r.note)}">阅读源码笔记 →</a><a href="${r.url}/tree/${r.commit}" target="_blank" rel="noopener noreferrer">固定源码 ↗</a></div><small class="source-version">${r.commit.slice(0,10)} · 局部源码审读</small></article>`).join('')}</div><div class="notice neutral">固定源码表示有版本记录和局部审读，完整运行与兼容性需要分别验证。<a href="${readLink('RESOURCE_ATLAS.md')}">查看更多外部一手资源及公开范围</a>。</div>`;
}

function getQuiz(){return [
  {q:t('三个 logits 都为 0，正确标签为 B，交叉熵是多少？'),options:['0','ln(3) ≈ 1.0986','3'],answer:1,why:t('三个类别的概率均为 1/3，因此 loss = −ln(1/3) = ln(3)。')},
  {q:t('logits 为 [2, 1, 0]，正确标签为 A，梯度的符号是什么？'),options:['[−, +, +]','[+, −, +]','[+, +, −]'],answer:0,why:t('梯度等于 p − onehot(A)。正确类别的梯度为负，梯度下降会提高它的 logit。')},
  {q:t('两组有 1 和 3 个有效目标，局部平均 loss 为 0.4 和 2.4，全局 token 平均是多少？'),options:['1.4','1.9','2.8'],answer:1,why:t('(1 × 0.4 + 3 × 2.4) / 4 = 1.9。直接平均两组得到 1.4，表达的是另一个加权目标。')},
  {q:t('一个 token 的 loss mask 为 0，是否意味着改变它作为输入上下文也不会影响模型？'),options:[t('是，它完全被模型忽略'),t('否，它仍可能影响后续预测')],answer:1,why:t('忽略自身目标 loss 不等于屏蔽 attention；输入上下文仍可能改变后续预测。')}
];}
function renderLab(){
  main.innerHTML=heading(t('Interactive lab / 动手理解'),t('一枚 token，怎样改变参数？'),t('调整 logits，观察概率、loss 与梯度，再比较不同分组下的训练目标。'),ui`<a class="button" href="${readLink('lessons/01-one-token-to-update.md')}">阅读完整首课 →</a>`)+
  ui`<div class="lab-grid"><section class="panel"><span class="tag blue">01 · 输入</span><h2 style="margin-top:14px">三个候选 token</h2><p class="intro">Logit 是模型给每个候选的原始分数。</p>${['A','B','C'].map((name,i)=>ui`<div class="logit-control"><label for="logit-${i}">${name}</label><input type="range" id="logit-range-${i}" data-logit-range="${i}" min="-10" max="10" step="0.1" value="${Math.max(-10,Math.min(10,logits[i]))}" aria-label="Token ${name} logit 滑杆"><input id="logit-${i}" data-logit="${i}" type="number" min="-10000" max="10000" step="0.1" value="${logits[i]}" aria-label="Token ${name} logit 数值"></div>`).join('')}<label class="field"><span>正确标签</span><select id="target">${['A','B','C'].map((name,i)=>`<option value="${i}" ${target===i?'selected':''}>${name}</option>`).join('')}</select></label><label class="field"><span>学习率 η</span><input id="learning-rate" type="number" min="0" max="10" step="0.01" value="${rate}"></label><div class="lab-actions"><button class="button primary" data-action="step">执行一步更新 →</button><button class="button" data-action="shift">全部加 1000</button><button class="button ghost" data-action="reset-lab">恢复默认</button></div><p class="muted small-text" style="margin:16px 0 0">数值框支持更大范围；“全部加 1000”可检验概率对整体平移的不变性。</p><p id="lab-error" class="error small-text" role="status"></p></section><section class="panel" aria-label="计算结果"><span class="tag blue">02 · 从概率到梯度</span><div id="token-results">${tokenResultsShell()}</div><p id="lab-status" class="sr-only" role="status"></p></section></div>
  <div class="notice neutral">教学模型直接把三个 logits 当作参数，使用解析梯度。它验证局部数学关系，没有实现 Transformer、autograd 或真实分布式训练。</div>
  <section class="panel"><span class="tag blue">03 · 一个贯穿数据与分布式的问题</span><h2 style="margin-top:14px">分组变了，目标也会变吗？</h2><p class="intro">六个位置共享上面的 logits。标签依次为 A、C、C、C、B、A；勾选参与 loss 的目标。</p><div class="mask-list">${['A','C','C','C','B','A'].map((label,i)=>`<label><input type="checkbox" data-mask="${i}" ${masks[i]?'checked':''}>${i+1} · ${label}</label>`).join('')}</div><div class="batch-controls"><label for="partition">分组方式</label><select id="partition"><option value="uneven" ${partition==='uneven'?'selected':''}>位置 1 / 位置 2–6（默认有效数 1:3）</option><option value="balanced" ${partition==='balanced'?'selected':''}>位置 1–2 / 位置 3–6（默认有效数 2:2）</option><option value="empty" ${partition==='empty'?'selected':''}>位置 1–4 / 位置 5–6（默认有效数 4:0）</option></select></div><div id="batch-results"></div><p class="muted small-text">这里约定目标是“全局有效 token 平均”。有意让各组等权也可以，但它定义了另一个目标。mask 为 0 仅表示不计该目标的 loss。</p></section>
  <div class="section-title"><h2>先预测，再核对</h2><span class="tag">4 道自测 · 可重试</span></div><form id="quiz-form" class="panel"><div class="quiz-list">${getQuiz().map((q,i)=>`<div class="quiz-item"><fieldset><legend>${i+1}. ${q.q}</legend><div class="quiz-options">${q.options.map((option,j)=>`<label><input type="radio" name="quiz-${i}" value="${j}" required>${option}</label>`).join('')}</div><p class="quiz-feedback" id="quiz-feedback-${i}" hidden></p></fieldset></div>`).join('')}</div><button class="button primary" type="submit">核对答案</button><p id="quiz-score" class="small-text" role="status" style="margin:15px 0 0"></p><p class="muted small-text" style="margin:12px 0 0">自测帮助检查理解，不会自动标记模块结业。解释机制、完成练习并保留证据后，再自主验收。</p></form>`;
  syncLogitInputs();updateLab();
}
function fixed(value,digits=6){return Number(value).toFixed(digits);}
function updateLab(){
  const error=document.getElementById('lab-error');if(!error)return;
  try{
    const result=tokenStep(logits,target,rate);error.textContent='';
    updateTokenResults(document.getElementById('token-results'),result,target,lastStep,lossHistory);
    const batch=batchComparison(logits,masks,partition);
    const detailsOpen=main.querySelector('#batch-results details')?.open;
    document.getElementById('batch-results').innerHTML=batchResultsHtml(batch,masks);
    if(detailsOpen&&main.querySelector('#batch-results details'))main.querySelector('#batch-results details').open=true;
  }catch(e){error.textContent=e.message;document.getElementById('token-values').hidden=true;document.getElementById('batch-results').innerHTML='';}
}
function clearExperimentHistory(){lastStep=null;lossHistory=[];}
function announceLab(message){const status=document.getElementById('lab-status');if(status)status.textContent=message;}

function renderProgress(){
  const records=data.modules.filter(m=>entry(m.id).status!=='new'||entry(m.id).note||entry(m.id).evidence);
  const done=records.filter(m=>entry(m.id).status==='reviewed').length,learning=records.filter(m=>entry(m.id).status==='learning').length;
  main.innerHTML=heading(t('My learning / 个人记录'),t('把理解，变成可检查的能力。'),t('记录机制解释、代码实现、实验结果与未解决问题。进度由你自主验收。'))+
  ui`<div class="progress-overview"><div class="progress-stat"><strong>${learning}</strong><span>学习中</span></div><div class="progress-stat"><strong>${done}<small> / 16</small></strong><span>已自主验收</span></div><div class="progress-stat"><strong>${records.filter(m=>entry(m.id).note||entry(m.id).evidence).length}</strong><span>有记录的模块</span></div></div><div class="notice ${storageAvailable?'neutral':'warning'}">${storageAvailable?t('记录仅保存在当前浏览器，不会上传到 GitHub。清理浏览器数据或换设备时不会自动同步，请导出备份。'):t('当前浏览器无法持久保存，请及时导出记录。')}</div><div class="filter-bar"><button class="button primary" data-action="export-progress">导出学习记录 ↓</button><button class="button" data-action="import-progress">导入并合并记录 ↑</button><input type="file" id="progress-file" accept=".json,application/json" hidden><a class="button ghost" href="${readLink('templates/learner-progress.md')}">Markdown 记录模板 ↗</a></div><p class="muted small-text">导入会保留其他模块，并更新文件中同名模块的记录；仅导入你信任的个人备份。</p>
  ${records.length?records.map(m=>{const r=entry(m.id);return `<article class="progress-entry"><header><h3><a href="${moduleLink(m.id)}"><span class="mono">${m.id}</span> ${escape(m.title)} →</a></h3>${statusTag(m.id)}</header>${r.note?`<p class="saved-note">${escape(r.note)}</p>`:''}${r.evidence?ui`<details><summary>验收证据</summary><p class="saved-note">${escape(r.evidence)}</p></details>`:''}${r.updated?ui`<time datetime="${escape(r.updated)}">更新：${escape(new Date(r.updated).toLocaleString(getLocale()==='en'?'en-US':'zh-CN'))}</time>`:''}</article>`;}).join(''):t('<div class="empty-state"><h2>从一个具体问题开始。</h2><p>进入模块，写下第一条理解或疑问。记录会出现在这里。</p><a class="button primary" href="#/module/M00">开始 M00 →</a></div>')}`;
}
function renderSearch(query){
  const results=searchDocuments(data.docs,query);
  main.innerHTML=heading(t('Search / 全文检索'),query?t('搜索「')+query+t('」'):t('搜索学习资料'),query?ui`找到 ${results.length} 篇相关资料。支持中文、英文、模块编号与代码标识。`:t('在顶部输入想理解的概念，例如 loss mask、checkpoint 或 GRPO。'))+
  (results.length?results.map(r=>`<a class="search-result" href="${readLink(r.path)}"><small>${escape(r.path)}</small><h2>${escape(r.title)}</h2><p>${escape(r.snippet)}…</p></a>`).join(''):ui`<div class="empty-state"><p>${query?t('暂时没有匹配内容。试试更短的关键词或英文术语。'):t('从一个具体概念开始搜索。')}</p><a href="#/resources">浏览资源与手册 →</a></div>`);
}
function renderMissing(){main.innerHTML=heading('Page not found',t('这份内容暂时没有找到。'),t('链接可能已更新。可以回到学习路径，或搜索文档标题。'), t('<a class="button primary" href="#/learn">返回学习路径 →</a>'));}

async function navigate({preserve=null}={}){
  if(!data)return;
  readerCleanup();readerCleanup=()=>{};mapController?.disconnect();mapController=null;
  const version=++routeVersion;
  if(preserve)main.replaceChildren();
  const route=location.hash.slice(2)||'learn';const [pathname,query='']=route.split('?');const [name,...parts]=pathname.split('/');const params=new URLSearchParams(query);
  let file;try{file=decodeURIComponent(parts.join('/'));}catch{return renderMissing();}
  document.getElementById('current-section').textContent=titleNames()[name]||t('学习手册');
  document.title=(name==='module'?moduleById(file)?.title:name==='read'?(Object.hasOwn(data.docs,file)?data.docs[file].title:t('未找到内容')):titleNames()[name])+' · Frontier LLM Handbook';
  document.querySelectorAll('[data-nav]').forEach(a=>{const active=a.dataset.nav===(name==='module'?'learn':name==='read'?'resources':name);a.classList.toggle('active',active);if(active)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});
  document.getElementById('sidebar').classList.remove('open');document.getElementById('menu-toggle').setAttribute('aria-expanded','false');
  if(name==='learn')renderLearn();else if(name==='module')renderModule(file);else if(name==='read')renderDocument(file,params.get('anchor'));else if(name==='map')renderMap();else if(name==='resources')renderResources();else if(name==='lab')renderLab();else if(name==='progress')renderProgress();else if(name==='search')renderSearch(params.get('q')||'');else renderMissing();
  updateSidebar();
  if(preserve)restoreForm(preserve);else{window.scrollTo(0,0);main.focus({preventScroll:true});}
  readerCleanup=observeReading(main);
  if(!preserve&&!matchMedia('(prefers-reduced-motion: reduce)').matches)main.animate([{opacity:.55,transform:'translateY(7px)'},{opacity:1,transform:'translateY(0)'}],{duration:260,easing:'cubic-bezier(.22,1,.36,1)'});
  await renderMermaid(version);
  if(version!==routeVersion)return;
  if(preserve){
    restoreDetails(preserve);
    const h=main.querySelectorAll('.markdown h1,.markdown h2,.markdown h3,.markdown h4')[preserve.headingIndex];
    window.scrollTo(0,h?h.getBoundingClientRect().top+window.scrollY-preserve.headingOffset:preserve.scrollY);
  }else if(name==='read'&&params.get('anchor'))scrollHeading(params.get('anchor'));
}

function localizeShell(){
  document.documentElement.lang=getLocale()==='en'?'en':'zh-CN';
  document.documentElement.dataset.language=getLocale();
  document.querySelectorAll('[data-i18n]').forEach(el=>el.textContent=t(el.dataset.i18n));
  for(const attr of ['aria-label','placeholder','content'])document.querySelectorAll(`[data-i18n-${attr}]`).forEach(el=>el.setAttribute(attr,t(el.getAttribute('data-i18n-'+attr))));
  document.getElementById('language-select').value=getLocale();
}
async function loadContent(language){
  if(contentCache.has(language))return contentCache.get(language);
  const response=await fetch(new URL(language==='en'?'./content.en.json':'./content.json',document.baseURI));
  if(!response.ok)throw new Error(t('资料请求失败 ')+response.status);
  const content=await response.json();
  if(content.schemaVersion!==1||content.modules?.length!==16||content.locale!==language)throw new Error(t('资料格式不兼容'));
  contentCache.set(language,content);return content;
}
function captureForm(){
  const headings=[...main.querySelectorAll('.markdown h1,.markdown h2,.markdown h3,.markdown h4')];
  let headingIndex=-1;headings.forEach((h,i)=>{if(h.getBoundingClientRect().top<=140)headingIndex=i;});
  return {
    fields:[...main.querySelectorAll('input:not([type=file]),select,textarea')].map(el=>({id:el.id,name:el.name,type:el.type,value:el.value,checked:el.checked})),
    details:pageDetails().map(({key,el})=>({key,open:el.open})),
    quizChecked:Boolean(document.getElementById('quiz-score')?.textContent),
    headingIndex,headingOffset:headings[headingIndex]?.getBoundingClientRect().top||0,scrollY:window.scrollY
  };
}
function pageDetails(){
  let regular=0;return [...main.querySelectorAll('details')].map(el=>({el,key:el.dataset.diagramIndex===undefined?'detail:'+regular++:'diagram:'+el.dataset.diagramIndex}));
}
function restoreDetails(snapshot){
  const states=new Map(snapshot.details.map(item=>[item.key,item.open]));
  pageDetails().forEach(({key,el})=>{if(states.has(key))el.open=states.get(key);});
}
function restoreForm(snapshot){
  for(const field of snapshot.fields){
    const el=field.id?document.getElementById(field.id):main.querySelector(`[name="${CSS.escape(field.name)}"][value="${CSS.escape(field.value)}"]`);
    if(!el||!main.contains(el))continue;
    if(field.type==='radio'||field.type==='checkbox')el.checked=field.checked;else el.value=field.value;
  }
  if(snapshot.quizChecked)checkQuiz(document.getElementById('quiz-form'));
}
async function changeLanguage(next){
  if(next===getLocale())return;
  const select=document.getElementById('language-select');select.disabled=true;
  try{
    const content=await loadContent(next),snapshot=captureForm();
    setLocale(next);data=content;
    try{localStorage.setItem(languageKey,next);}catch{}
    const url=new URL(location.href);url.searchParams.set('lang',next);history.replaceState(null,'',url);
    localizeShell();refreshTheme();document.getElementById('toast').hidden=true;
    await navigate({preserve:snapshot});
  }catch{select.value=getLocale();toast(t('语言版本暂时无法加载，请稍后重试。'));}
  finally{select.disabled=false;}
}

function saveModule(id,explicit=true){
  const status=document.getElementById('module-status').value,note=document.getElementById('module-note').value,evidence=document.getElementById('module-evidence').value;
  if(status==='reviewed'&&!evidence.trim()){toast(t('请先填写验收证据，再标记为已自主验收。'));document.getElementById('module-evidence').focus();return;}
  progress.modules[id]={status,note,evidence,updated:new Date().toISOString()};
  const saved=saveProgress();updateSidebar();
  document.getElementById('save-feedback').textContent=saved?t('已保存到当前浏览器'):t('暂存于本次页面，请导出备份');
  if(explicit)toast(saved?t('学习记录已保存。'):t('请在“我的学习”导出本次记录。'));
}
function syncLogitInputs(){logits.forEach((value,i)=>{document.getElementById('logit-'+i).value=String(value);const range=document.getElementById('logit-range-'+i);range.value=String(Math.max(-10,Math.min(10,value)));range.disabled=Math.abs(value)>10;});}

main.addEventListener('click',async event=>{
  const el=event.target.closest('button');if(!el)return;
  if(el.dataset.stage){stageFilter=el.dataset.stage;focusPreservingRender(renderLearn);return;}
  if(el.dataset.resourceFilter){resourceFilter=el.dataset.resourceFilter;focusPreservingRender(renderResources);return;}
  if(el.dataset.mapMode){mapMode=el.dataset.mapMode;focusPreservingRender(renderMap);return;}
  if(el.dataset.connection){connectionId=el.dataset.connection;focusPreservingRender(renderMap);return;}
  if(el.dataset.mapNode){selectedModule=el.dataset.mapNode;if(mapMode==='knowledge'){const c=data.connections.find(c=>c.modules.includes(selectedModule));if(c)connectionId=c.id;else{mapMode='prerequisite';}}focusPreservingRender(renderMap);return;}
  if(el.dataset.heading){scrollHeading(el.dataset.heading);return;}
  const action=el.dataset.action;
  if(action==='save-module')saveModule(el.dataset.id);
  if(action==='step'){
    try{const before=tokenStep(logits,target,rate);if(before.next.some(v=>Math.abs(v)>10000)){toast(t('下一步超出输入范围，请先调整 logits 或降低学习率。'));return;}if(!lossHistory.length)lossHistory.push(before.loss);lossHistory.push(before.nextLoss);lastStep={before:before.loss,after:before.nextLoss,probabilities:[...before.probabilities]};logits=before.next;syncLogitInputs();updateLab();announceLab(t('完成一次更新，loss 从 ')+fixed(before.loss)+t(' 变为 ')+fixed(before.nextLoss));}catch(e){toast(e.message);}
  }
  if(action==='shift'){
    if(logits.some(v=>!Number.isFinite(v)||Math.abs(v+1000)>10000)){toast(t('平移后超出教学范围，请先调整 logits。'));return;}
    logits=logits.map(v=>v+1000);clearExperimentHistory();syncLogitInputs();updateLab();
  }
  if(action==='reset-lab'){logits=[2,1,0];target=1;rate=.1;masks=[1,1,1,1,0,0];partition='uneven';clearExperimentHistory();syncLogitInputs();document.getElementById('target').value='1';document.getElementById('learning-rate').value='0.1';document.getElementById('partition').value='uneven';main.querySelectorAll('[data-mask]').forEach(el=>el.checked=Boolean(masks[Number(el.dataset.mask)]));updateLab();announceLab(t('实验参数已恢复默认，自测作答保持不变。'));}
  if(action==='export-progress'){
    const blob=new Blob([JSON.stringify(progress,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download='frontier-llm-learning-'+new Date().toISOString().slice(0,10)+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  if(action==='import-progress')document.getElementById('progress-file').click();
});

main.addEventListener('input',event=>{
  const el=event.target;
  if(el.dataset.logit!==undefined){logits[Number(el.dataset.logit)]=el.value.trim()===''?NaN:Number(el.value);clearExperimentHistory();const range=document.getElementById('logit-range-'+el.dataset.logit);range.value=String(Math.max(-10,Math.min(10,logits[Number(el.dataset.logit)])));range.disabled=Math.abs(logits[Number(el.dataset.logit)])>10;updateLab();}
  if(el.dataset.logitRange!==undefined){logits[Number(el.dataset.logitRange)]=Number(el.value);document.getElementById('logit-'+el.dataset.logitRange).value=el.value;clearExperimentHistory();updateLab();}
  if(el.id==='learning-rate'){rate=el.value.trim()===''?NaN:Number(el.value);clearExperimentHistory();updateLab();}
  if(el.id==='module-note'||el.id==='module-evidence'){
    const id=el.dataset.module,old=entry(id);
    progress.modules[id]={...old,[el.id==='module-note'?'note':'evidence']:el.value,updated:new Date().toISOString()};
    if(progress.modules[id].status==='reviewed'&&!progress.modules[id].evidence.trim()){progress.modules[id].status='learning';document.getElementById('module-status').value='learning';}
    const saved=saveProgress();document.getElementById('save-feedback').textContent=saved?t('笔记已自动保存；状态修改请点保存'):t('暂存于本次页面，请导出备份');updateSidebar();
  }
});
main.addEventListener('change',async event=>{
  const el=event.target;
  if(el.id==='target'){target=Number(el.value);clearExperimentHistory();updateLab();}
  if(el.dataset.mask!==undefined){masks[Number(el.dataset.mask)]=Number(el.checked);updateLab();announceLab(t('当前有 ')+masks.reduce((a,b)=>a+b,0)+t(' 个有效目标。'));}
  if(el.id==='partition'){partition=el.value;updateLab();announceLab(t('已切换分组，结果已更新。'));}
  if(el.id==='progress-file'&&el.files?.[0]){
    try{if(el.files[0].size>1000000)throw new Error(t('备份文件过大，请选择本站导出的学习记录。'));const incoming=validateProgress(JSON.parse(await el.files[0].text()));progress={...progress,modules:{...progress.modules,...incoming.modules},lastModule:incoming.lastModule||progress.lastModule};const saved=saveProgress();renderProgress();updateSidebar();toast(saved?t('学习记录已导入并合并。'):t('记录已导入本次页面，请及时导出备份。'));}catch(e){toast(t('导入失败：')+e.message);}
  }
});
function checkQuiz(form){
  if(!form)return;const values=new FormData(form);let correct=0;
  getQuiz().forEach((q,i)=>{const ok=Number(values.get('quiz-'+i))===q.answer;correct+=Number(ok);const output=document.getElementById('quiz-feedback-'+i);output.hidden=false;output.className='quiz-feedback '+(ok?'good':'error');output.textContent=(ok?t('✓ 正确。'):t('再想一想。'))+q.why;});
  document.getElementById('quiz-score').textContent=ui`本次答对 ${correct} / 4。${correct===4?t('接下来试着不看公式解释机制，并完成模块练习。'):t('对照解释修改答案，可以再次核对。')}`;
}
main.addEventListener('submit',event=>{if(event.target.id==='quiz-form'){event.preventDefault();checkQuiz(event.target);}});
mountIcons();
localizeShell();
const refreshTheme=initializeTheme(()=>{void renderMermaid(routeVersion);});
document.getElementById('language-select').addEventListener('change',event=>{void changeLanguage(event.target.value);});
document.getElementById('search-form').addEventListener('submit',event=>{event.preventDefault();location.hash='#/search?q='+encodeURIComponent(document.getElementById('global-search').value.trim());});
document.getElementById('menu-toggle').addEventListener('click',()=>{const open=document.getElementById('sidebar').classList.toggle('open');document.getElementById('menu-toggle').setAttribute('aria-expanded',String(open));});
document.querySelector('.skip-link').addEventListener('click',event=>{event.preventDefault();main.focus({preventScroll:true});window.scrollTo(0,0);});
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
      document.getElementById('save-feedback').textContent=t('已同步另一页面的最新记录');
      toast(t('另一页面更新了本模块，已同步最新记录。'));
    }
    if(location.hash==='#/progress')renderProgress();
  }catch{toast(t('另一页面的记录格式无法识别，请先导出当前记录。'));}
});

try{const saved=localStorage.getItem(storageKey);if(saved)progress=validateProgress(JSON.parse(saved));}catch{storageAvailable=false;toast(t('未能读取浏览器中的学习记录，请使用备份导入。'));}
try{
  data=await loadContent(getLocale());
  await navigate();
}catch(error){main.innerHTML=heading('Loading error',t('学习资料暂时无法加载。'),t('请刷新页面重试，或先在 GitHub 阅读原文。'),ui`<a class="button primary" href="https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/ROADMAP.md">阅读完整学习路线 ↗</a>`);console.error(error);}
finally{document.getElementById('language-select').disabled=false;}

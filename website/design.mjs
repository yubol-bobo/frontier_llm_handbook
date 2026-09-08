import {t,ui} from './i18n.mjs';
import {Route,Network,Library,FlaskConical,NotebookPen,ArrowUpRight,ArrowRight,BookOpen,Check,Search,Menu,Sun,Moon,Layers,GitBranch,Play,ChevronRight,ArrowDownRight,Code,FileText,Activity,Link} from 'lucide';

const icons={route:Route,network:Network,library:Library,flask:FlaskConical,notebook:NotebookPen,'arrow-up-right':ArrowUpRight,arrow:ArrowRight,book:BookOpen,check:Check,search:Search,menu:Menu,sun:Sun,moon:Moon,layers:Layers,git:GitBranch,play:Play,chevron:ChevronRight,'arrow-down-right':ArrowDownRight,code:Code,file:FileText,activity:Activity,link:Link};
export function icon(name,className=''){
  const node=icons[name]||ArrowRight;
  return `<svg class="icon ${className}" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${node.map(([tag,attrs])=>`<${tag} ${Object.entries(attrs).map(([key,value])=>`${key}="${value}"`).join(' ')}/>`).join('')}</svg>`;
}
export function mountIcons(root=document){root.querySelectorAll('[data-icon]').forEach(el=>{el.innerHTML=icon(el.dataset.icon);});}
export const escapeText=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

export function renderLearningView({data,progress,entry,readLink,moduleLink,statusTag,stageFilter}){
  const started=progress.lastModule&&entry(progress.lastModule).status!=='reviewed'?data.modules.find(m=>m.id===progress.lastModule):null;
  const next=started||data.modules.find(m=>entry(m.id).status!=='reviewed')||data.modules[0];
  const stageNames=['FOUNDATIONS','SYSTEMS','POST-TRAINING'];
  const moduleCard=m=>`<a class="module-card" data-stage-color="${m.stage}" href="${moduleLink(m.id)}"><div class="module-card-top"><span class="module-code">${m.id}</span>${statusTag(m.id)}</div><h3>${escapeText(m.title)}</h3><p>${escapeText(m.question)}</p><div class="module-card-bottom"><span>${icon('book')} ${escapeText(m.materials)}</span><span class="card-arrow">${icon('arrow-up-right')}</span></div></a>`;
  return ui`<div class="page-heading learn-heading"><div><div class="eyebrow"><span class="eyebrow-dot"></span> THE FRONTIER IS OPEN</div><h1>理解原理。<span>构建前沿。</span></h1><p>从第一枚 token，到完整训练系统。<br class="mobile-break">一条有源码、有实践、有连接的学习路径。</p></div><a class="text-link" href="${readLink('ROADMAP.md')}">完整路线与入门诊断 ${icon('arrow-up-right')}</a></div>
  <div class="learn-overview"><section class="continue-card"><div class="continue-content"><div class="eyebrow">${icon('play')}${started?t('CONTINUE LEARNING / 继续学习'):t('YOUR FIRST STEP / 从这里开始')}</div><span class="continue-module">${next.id}<span>CORE MODULE</span></span><h2>${escapeText(next.title)}</h2><p>${escapeText(next.deliverable)}</p><a class="button primary" href="${moduleLink(next.id)}">${started?t('继续探索'):t('进入学习')} ${icon('arrow')}</a></div><div class="continue-route" aria-label="三个学习阶段">${data.stages.map((s,i)=>`<a class="mini-route ${next.stage===s.id?'current':''}" href="${moduleLink(s.ids[0])}"><span class="mini-route-dot">${next.stage===s.id?icon('play'):'0'+(i+1)}</span><span><small>${stageNames[i]}</small><strong>${s.name}</strong></span>${icon('chevron')}</a>`).join('')}</div></section>
  <a class="experiment-teaser" href="#/lab"><div class="teaser-top"><span class="tag mint">${icon('flask')} INTERACTIVE LAB</span>${icon('arrow-up-right')}</div><div class="mini-probabilities" aria-label="默认示例中 token A、B、C 的概率"><div><span>A</span><i style="--prob:.66524"></i><small>66.52%</small></div><div class="target"><span>B</span><i style="--prob:.24473"></i><small>24.47%</small></div><div><span>C</span><i style="--prob:.09003"></i><small>9.00%</small></div></div><h3>让公式动起来。</h3><p>改变一个 logit，观察概率与梯度。</p><span class="text-link">打开训练实验 ${icon('arrow')}</span></a></div>
  <div class="learning-meta"><span><strong>16</strong> 核心模块</span><span><strong>06</strong> 前沿专题</span><span><strong>${data.resources.length}</strong> 固定源码项目</span><span class="meta-note">${icon('code')} 普通电脑即可开始</span></div>
  <div class="section-title curriculum-title"><div><span class="eyebrow">THE CURRICULUM</span><h2>循序深入，连接全局。</h2></div><button class="chip ${stageFilter==='all'?'active':''}" data-stage="all" aria-pressed="${stageFilter==='all'}">全部阶段 ${icon('layers')}</button></div>
  <div class="stage-switcher" aria-label="选择学习阶段">${data.stages.map((s,i)=>`<button class="stage-choice ${stageFilter===s.id?'active':''}" data-stage="${s.id}" data-stage-color="${s.id}" aria-pressed="${stageFilter===s.id}"><span class="stage-choice-index">0${i+1}</span><span><small>${stageNames[i]}</small><strong>${s.name}</strong></span><span class="stage-choice-count">${s.ids.filter(id=>entry(id).status==='reviewed').length}/${s.ids.length}</span></button>`).join('')}</div>
  <div class="curriculum-stages">${data.stages.filter(s=>stageFilter==='all'||s.id===stageFilter).map(s=>`<section class="stage" data-stage-color="${s.id}"><div class="stage-heading"><span class="stage-line-dot"></span><h2>${s.name}</h2><span class="mono">${s.range}</span><span class="stage-description">${s.description}</span></div><div class="module-list">${data.modules.filter(m=>m.stage===s.id).map(moduleCard).join('')}</div></section>`).join('')}</div>
  <div class="notice neutral">每个模块包含先修、阅读、练习与验收。评估贯穿全程；<a href="${readLink('ROADMAP.md')}">已有基础可按诊断进入专项路线 ${icon('arrow-up-right')}</a>。</div>
  <div class="section-title"><div><span class="eyebrow">BEYOND THE CORE</span><h2>下一站，未完成的前沿。</h2></div><span class="muted small-text">按兴趣选读</span></div><div class="seminar-grid">${data.seminars.map(s=>`<a class="seminar" href="${readLink(s.file,s.anchor)}"><span>${s.id}</span><strong>${escapeText(s.title)}</strong>${icon('arrow-up-right')}</a>`).join('')}</div>`;
}

export function focusPreservingRender(render){
  const active=document.activeElement;
  const attribute=['data-stage','data-resource-filter','data-map-mode','data-map-node','data-connection','data-action'].find(key=>active?.hasAttribute(key));
  const value=attribute?active.getAttribute(attribute):null;
  render();
  if(attribute)document.querySelector(`[${attribute}="${CSS.escape(value)}"]`)?.focus({preventScroll:true});
}

export function relationEdges(mode,prerequisites,selected){
  return mode==='prerequisite'?prerequisites.map(([from,to])=>({from,to,directed:true})):selected.modules.map(to=>({from:'topic',to,directed:false}));
}

export function curveBetween(a,b,bend=65){
  if(Math.abs(a.x-b.x)<40)return `M ${a.x} ${a.y} C ${a.x+bend} ${a.y}, ${b.x+bend} ${b.y}, ${b.x} ${b.y}`;
  const mid=(a.x+b.x)/2;
  return `M ${a.x} ${a.y} C ${mid} ${a.y}, ${mid} ${b.y}, ${b.x} ${b.y}`;
}

export function lossHistoryPoints(values,width=340,height=82){
  if(values.some(v=>!Number.isFinite(v)))throw new Error('Loss history must be finite');
  if(!values.length)return '';
  const low=Math.min(...values),high=Math.max(...values),span=Math.max(high-low,.000001);
  return values.map((v,i)=>`${8+(values.length===1?0:i/(values.length-1))*(width-16)},${8+(1-(v-low)/span)*(height-16)}`).join(' ');
}

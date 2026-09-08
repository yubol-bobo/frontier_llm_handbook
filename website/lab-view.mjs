import {t,ui} from './i18n.mjs';
import {escapeText,lossHistoryPoints} from './design.mjs';
const fixed=(value,digits=6)=>Number(value).toFixed(digits);

export function tokenResultsShell(){
  return ui`<div id="token-values"><div class="probabilities">${['A','B','C'].map((name,i)=>ui`<div class="prob-row" data-prob-row="${i}"><span class="prob-label">${name}</span><div class="prob-track" role="meter" aria-label="Token ${name} 概率" aria-valuemin="0" aria-valuemax="1" aria-valuenow="0"><div class="prob-bar" style="transform:scaleX(0)"></div><span class="prob-ghost" hidden></span></div><span class="prob-value"></span></div>`).join('')}</div><div class="prob-legend" id="prob-legend" hidden>细线标记上一步更新前的概率</div><div class="metric-row"><div><small>当前交叉熵 loss</small><strong id="current-loss"></strong></div><div><small>下一步预期 loss</small><strong id="predicted-loss"></strong></div></div><div class="math-line">gradient = p − onehot(y)</div><details class="precision-details"><summary>查看精确梯度与参数</summary><div class="table-scroll"><table class="data-table"><thead><tr><th>Token</th><th>梯度</th><th>下一步 logit</th></tr></thead><tbody>${['A','B','C'].map((name,i)=>`<tr><td>${name}</td><td class="mono" data-gradient="${i}"></td><td class="mono" data-next="${i}"></td></tr>`).join('')}</tbody></table></div></details><p class="muted small-text" style="margin-top:18px">z′ = z − η × gradient。正梯度让对应 logit 降低，负梯度让它提高。</p><div id="step-feedback" class="step-result" hidden></div><div id="loss-history" class="loss-history" hidden></div></div>`;
}

export function updateTokenResults(root,result,target,lastStep,history){
  root.querySelector('#token-values').hidden=false;
  for(let i=0;i<3;i++){
    const row=root.querySelector(`[data-prob-row="${i}"]`),p=result.probabilities[i];
    row.classList.toggle('target',i===target);row.querySelector('.prob-label').textContent=['A','B','C'][i]+(i===target?' ✓':'');
    row.querySelector('.prob-bar').style.transform=`scaleX(${p})`;
    row.querySelector('[role="meter"]').setAttribute('aria-valuenow',String(p));row.querySelector('.prob-value').textContent=(p*100).toFixed(2)+'%';
    const ghost=row.querySelector('.prob-ghost');ghost.hidden=!lastStep;if(lastStep)ghost.style.left=(lastStep.probabilities[i]*100)+'%';
    root.querySelector(`[data-gradient="${i}"]`).textContent=(result.gradient[i]>=0?'+':'')+fixed(result.gradient[i]);root.querySelector(`[data-next="${i}"]`).textContent=fixed(result.next[i]);
  }
  root.querySelector('#prob-legend').hidden=!lastStep;
  root.querySelector('#current-loss').textContent=fixed(result.loss);root.querySelector('#predicted-loss').textContent=fixed(result.nextLoss);
  const feedback=root.querySelector('#step-feedback');feedback.hidden=!lastStep;
  if(lastStep){const delta=lastStep.after-lastStep.before;feedback.innerHTML=ui`<small>JUST UPDATED / 刚才这一步</small><div class="step-loss"><span>${fixed(lastStep.before)}</span><span>→</span><strong>${fixed(lastStep.after)}</strong></div><div class="step-delta">Loss ${delta<=0?t('降低'):t('升高')} ${fixed(Math.abs(delta))}</div>`;}
  const chart=root.querySelector('#loss-history');chart.hidden=history.length<2;
  if(history.length>1)chart.innerHTML=ui`<header><span>本次参数更新轨迹</span><span>${history.length-1} 次更新</span></header><svg viewBox="0 0 340 82" role="img" aria-label="Loss 从 ${escapeText(fixed(history[0]))} 变为 ${escapeText(fixed(history.at(-1)))}"><line class="chart-baseline" x1="8" y1="74" x2="332" y2="74"/><polyline points="${lossHistoryPoints(history)}"/></svg><p>纵轴自适应：${fixed(Math.min(...history),4)}–${fixed(Math.max(...history),4)}。修改输入条件后重新记录。</p>`;
}

export function batchResultsHtml(batch,masks){
  if(!batch.count)return t('<div class="notice warning">没有有效目标，无法定义平均 loss。请至少勾选一个目标；不能把空批次的 loss 记作 0。</div>');
  const different=Math.abs(batch.global.loss-batch.equalGroups.loss)>1e-8;
  return ui`<div class="batch-groups">${batch.groups.map((group,i)=>ui`<div class="batch-group"><header><span>GROUP ${i+1}</span><b>${group.count} 个有效目标</b></header><div class="batch-tokens">${group.ids.map(id=>ui`<span class="batch-token ${masks[id]?'':'masked'}" aria-label="位置 ${id+1}，标签 ${['A','C','C','C','B','A'][id]}，${masks[id]?t('计入 loss'):t('已屏蔽目标')}">${['A','C','C','C','B','A'][id]}</span>`).join('')}</div></div>`).join('')}</div><div class="aggregation-comparison"><div><small>全局 token 平均</small><strong>${fixed(batch.global.loss)}</strong></div><span>${different?'≠':'='}</span><div><small>局部均值直接平均</small><strong class="${different?'different':''}">${fixed(batch.equalGroups.loss)}</strong></div></div><p class="small-text muted">有效目标共 ${batch.count} 个，两组 ${batch.groups.map(g=>g.count).join(' : ')}。${batch.groups.some(g=>!g.count)?t('空组不参与平均，贡献为 0。'):''}</p><details class="precision-details"><summary>查看三种聚合的 loss 与梯度</summary><div class="table-scroll"><table class="data-table"><thead><tr><th>聚合规则</th><th>Loss</th><th>梯度 [A, B, C]</th></tr></thead><tbody>${[[t('全局 token 平均'),batch.global],[t('按有效数量加权'),batch.weighted],[t('局部均值直接平均'),batch.equalGroups]].map(([label,value])=>`<tr><td>${label}</td><td class="mono">${fixed(value.loss)}</td><td class="mono">[${value.gradient.map(g=>fixed(g,4)).join(', ')}]</td></tr>`).join('')}</tbody></table></div></details><div class="math-line">L = Σ (n_group × L_group) / Σ n_group</div><p class="${different?'error':'good'} small-text">${different?t('本例中，直接平均局部均值改变了目标。'):t('当前两种 loss 相同；是否是同一目标，还需比较权重与梯度。')}</p>`;
}

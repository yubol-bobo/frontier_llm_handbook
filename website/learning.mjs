export const statuses={new:'未开始',learning:'学习中',reviewed:'已自主验收'};
export function tokenStep(logits,target,rate=0.1) {
  if(logits.length!==3 || logits.some(v=>!Number.isFinite(v)||Math.abs(v)>10000) || !Number.isInteger(target)||target<0||target>2 || !Number.isFinite(rate)||rate<0||rate>10) throw new Error('请使用有限数值：logit 绝对值不超过 10000，学习率为 0–10。');
  const max=Math.max(...logits), exps=logits.map(v=>Math.exp(v-max)),sum=exps.reduce((a,b)=>a+b,0);
  const probabilities=exps.map(v=>v/sum),loss=(max-logits[target])+Math.log(sum);
  const gradient=probabilities.map((v,i)=>v-Number(i===target));
  const next=logits.map((v,i)=>v-rate*gradient[i]);
  const nextMax=Math.max(...next);
  const nextLoss=(nextMax-next[target])+Math.log(next.reduce((sum,v)=>sum+Math.exp(v-nextMax),0));
  return {probabilities,loss,gradient,next,nextLoss};
}
export const partitions={uneven:[[0],[1,2,3,4,5]],balanced:[[0,1],[2,3,4,5]],empty:[[0,1,2,3],[4,5]]};
export function batchComparison(logits,masks=[1,1,1,1,0,0],partition='uneven') {
  if(masks.length!==6 || masks.some(x=>x!==0&&x!==1)||!partitions[partition]) throw new Error('无效的 mask 或分组。');
  const targets=[0,2,2,2,1,0];
  const rows=targets.map(t=>tokenStep(logits,t,0));
  const count=masks.reduce((a,b)=>a+b,0);
  if(!count)return {count:0,groups:[],global:null,weighted:null,equalGroups:null};
  const groups=partitions[partition].map(ids=>{
    const active=ids.filter(i=>masks[i]),n=active.length;
    return {count:n,ids,loss:n?active.reduce((a,i)=>a+rows[i].loss,0)/n:null,gradient:n?[0,1,2].map(j=>active.reduce((a,i)=>a+rows[i].gradient[j],0)/n):null};
  });
  const active=groups.filter(g=>g.count);
  return {count,groups,global:{loss:rows.reduce((a,r,i)=>a+masks[i]*r.loss,0)/count,gradient:[0,1,2].map(j=>rows.reduce((a,r,i)=>a+masks[i]*r.gradient[j],0)/count)},weighted:{loss:active.reduce((a,g)=>a+g.count*g.loss,0)/count,gradient:[0,1,2].map(j=>active.reduce((a,g)=>a+g.count*g.gradient[j],0)/count)},equalGroups:{loss:active.reduce((a,g)=>a+g.loss,0)/active.length,gradient:[0,1,2].map(j=>active.reduce((a,g)=>a+g.gradient[j],0)/active.length)}};
}
export function emptyProgress(){return {schemaVersion:1,modules:{},lastModule:null};}
export function validateProgress(value) {
  if(!value || value.schemaVersion!==1 || typeof value.modules!=='object' || value.modules===null || Array.isArray(value.modules))throw new Error('无法识别学习记录格式。');
  const result=emptyProgress();
  for(const [id,item] of Object.entries(value.modules)) {
    if(!/^M(?:0\d|1[0-5])$/.test(id)||!item||!Object.hasOwn(statuses,item.status)||typeof item.note!=='string'||item.note.length>20000||typeof item.evidence!=='string'||item.evidence.length>5000)throw new Error('学习记录包含无效模块或内容。');
    if(item.status==='reviewed'&&!item.evidence.trim())throw new Error('自主验收需要保留证据说明。');
    result.modules[id]={status:item.status,note:item.note,evidence:item.evidence,updated:typeof item.updated==='string'?item.updated:''};
  }
  result.lastModule=/^M(?:0\d|1[0-5])$/.test(value.lastModule||'')?value.lastModule:null;
  return result;
}
export function searchDocuments(docs,query) {
  const words=query.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
  if(!words.length)return [];
  return Object.values(docs).map(doc=>{
    const title=doc.title.toLocaleLowerCase(),text=doc.markdown.toLocaleLowerCase();
    if(!words.every(word=>text.includes(word)||title.includes(word)))return null;
    const score=words.reduce((n,word)=>n+(title.includes(word)?12:0)+Math.min(text.split(word).length-1,8),0);
    const offset=Math.max(0,text.indexOf(words[0])-65);
    return {path:doc.path,title:doc.title,score,snippet:doc.markdown.slice(offset,offset+200).replace(/[#*`>|]/g,'')};
  }).filter(Boolean).sort((a,b)=>b.score-a.score);
}

import {icon,relationEdges,curveBetween} from './design.mjs';

export function initializeTheme(onChange=()=>{}){
  const key='frontier-llm-handbook:appearance:v1';
  const apply=theme=>{
    document.documentElement.dataset.theme=theme;
    const button=document.getElementById('theme-toggle');
    const label=theme==='dark'?'切换为浅色阅读主题':'切换为深色阅读主题';
    button.innerHTML=icon(theme==='dark'?'sun':'moon');
    button.setAttribute('aria-label',label);button.title=label;
    document.querySelector('meta[name="theme-color"]').content=theme==='dark'?'#0d0f16':'#f7f8fc';
  };
  let saved;try{saved=localStorage.getItem(key);}catch{}
  apply(saved==='light'?'light':'dark');
  document.getElementById('theme-toggle').addEventListener('click',()=>{
    const next=document.documentElement.dataset.theme==='dark'?'light':'dark';
    apply(next);try{localStorage.setItem(key,next);}catch{}onChange(next);
  });
}

export function observeReading(root){
  const article=root.querySelector('.markdown');
  const progress=document.getElementById('reading-progress');
  if(!article){progress.hidden=true;return ()=>{};}
  const headings=[...root.querySelectorAll('.toc [data-heading]')].map(button=>({button,target:document.getElementById(button.dataset.heading)})).filter(item=>item.target);
  progress.hidden=false;
  let frame=0;
  const update=()=>{
    frame=0;
    const offset=110;let active=headings[0];
    for(const item of headings)if(item.target.getBoundingClientRect().top<=offset)active=item;
    headings.forEach(item=>{const current=item===active;item.button.classList.toggle('active',current);if(current)item.button.setAttribute('aria-current','location');else item.button.removeAttribute('aria-current');});
    const rect=article.getBoundingClientRect(),distance=Math.max(1,rect.height-window.innerHeight+offset);
    const fraction=Math.max(0,Math.min(1,(offset-rect.top)/distance));
    progress.firstElementChild.style.transform=`scaleX(${fraction})`;
  };
  const request=()=>{if(!frame)frame=requestAnimationFrame(update);};
  window.addEventListener('scroll',request,{passive:true});window.addEventListener('resize',request);
  const observer=new ResizeObserver(request);observer.observe(article);request();
  return ()=>{window.removeEventListener('scroll',request);window.removeEventListener('resize',request);observer.disconnect();cancelAnimationFrame(frame);progress.hidden=true;};
}

export function mountMapWires(board,getState){
  let frame=0;
  const draw=()=>{
    frame=0;if(!board.isConnected)return;
    const svg=board.querySelector('.map-wires');if(!svg)return;
    const rect=board.getBoundingClientRect(),{mode,prerequisites,selected,selectedModule}=getState();
    const nodes=new Map([...board.querySelectorAll('[data-map-node]')].map(el=>[el.dataset.mapNode,el]));nodes.set('topic',board.querySelector('.map-topic'));
    svg.setAttribute('viewBox',`0 0 ${rect.width} ${rect.height}`);
    const edges=relationEdges(mode,prerequisites,selected);
    const marker='<defs><marker id="map-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M1 1L7 4L1 7" fill="none" stroke="var(--accent)" stroke-width="1.2"/></marker></defs>';
    svg.innerHTML=marker+edges.map(edge=>{
      const from=nodes.get(edge.from),to=nodes.get(edge.to);if(!from||!to)return '';
      const a=from.getBoundingClientRect(),b=to.getBoundingClientRect();
      let start,end,bend=65;
      if(edge.from==='topic'){start={x:a.left+a.width/2-rect.left,y:a.bottom-rect.top};end={x:b.left+b.width/2-rect.left,y:b.top-rect.top};}
      else if(Math.abs(a.left-b.left)<40){
        const leftSpace=Math.min(a.left,b.left)-rect.left,rightSpace=rect.right-Math.max(a.right,b.right);
        const right=rightSpace>=leftSpace;
        bend=(right?1:-1)*Math.max(0,Math.min(65,(right?rightSpace:leftSpace)-8));
        start={x:(right?a.right:a.left)-rect.left,y:a.top+a.height/2-rect.top};
        end={x:(right?b.right:b.left)-rect.left,y:b.top+b.height/2-rect.top};
      }
      else{const right=b.left>a.left;start={x:(right?a.right:a.left)-rect.left,y:a.top+a.height/2-rect.top};end={x:(right?b.left:b.right)-rect.left,y:b.top+b.height/2-rect.top};}
      const active=!edge.directed||edge.from===selectedModule||edge.to===selectedModule;
      return `<path class="map-wire ${active?'active':''} ${edge.directed?'':'knowledge'}" d="${curveBetween(start,end,bend)}" ${edge.directed&&active?'marker-end="url(#map-arrow)"':''}/>`;
    }).join('');
  };
  const request=()=>{if(!frame)frame=requestAnimationFrame(draw);};
  const observer=new ResizeObserver(request);observer.observe(board);board.querySelectorAll('[data-map-node]').forEach(node=>observer.observe(node));request();
  return {redraw:request,disconnect(){observer.disconnect();cancelAnimationFrame(frame);}};
}

import english from './locales/en-ui.json' with {type:'json'};

export const languageKey='frontier-llm-handbook:language:v1';
let locale='zh';
export const getLocale=()=>locale;
export function setLocale(value){if(value!=='zh'&&value!=='en')throw new Error('Unsupported language');locale=value;}
export function chooseLocale(url,saved){
  const requested=new URL(url).searchParams.get('lang');
  return requested==='en'||requested==='zh'?requested:saved==='en'?'en':'zh';
}
// Only call on authored UI strings. Never translate interpolated notes or queries.
export const messagePattern=()=>/[^<>"'\n]*\p{Script=Han}[^<>"'\n]*/gu;
export function t(source){
  if(locale==='zh')return source;
  return source.replace(messagePattern(),part=>Object.hasOwn(english,part)?english[part]:part)
    .replaceAll('。','. ').replaceAll('，',', ').replaceAll('；','; ').replaceAll('：',': ').replaceAll('「','“').replaceAll('」','”');
}
// Translate literal template segments before inserting dynamic (possibly personal) data.
export function ui(strings,...values){return strings.reduce((out,part,i)=>out+t(part)+(i<values.length?values[i]:''),'');}

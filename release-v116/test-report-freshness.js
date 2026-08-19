const fs=require('fs'),vm=require('vm');
global.window=global;
global.state={version:7,goals:{calories:2200,protein:180,carbs:225,fat:70,fiber:30,water:100},meals:[{id:'m1',date:'2026-08-18',name:'Eggs',type:'Breakfast',calories:300,protein:20,carbs:10,fat:15,fiber:1,source:'Manual',loggedAt:'2026-08-19T18:00:00Z'}],workouts:[],water:[],body:[],habits:{},healthHistory:{},aiReports:[{id:'old',type:'weekly',start:'2026-08-17',end:'2026-08-23',label:'Weekly',generatedAt:'2026-08-19T17:00:00Z',text:'old'}],reportSettings:{}};
function localISODate(d=new Date()){let y=d.getFullYear(),m=String(d.getMonth()+1).padStart(2,'0'),day=String(d.getDate()).padStart(2,'0');return `${y}-${m}-${day}`}
global.localISODate=localISODate;global.today=()=> '2026-08-19';global.num=x=>Number(x)||0;global.save=()=>{};global.render=()=>{};global.renderToday=()=>{};global.renderProgress=()=>{};global.renderSettings=()=>{};global.$=()=>null;global.native=()=>true;global.Android={hasApiKey:()=>true,generateReport:x=>{global.lastReq=JSON.parse(x)}};global.dateLabel=x=>x;global.uid=p=>p+'1';global.healthFor=()=>({});global.habitPct=()=>0;global.workoutSessions=()=>0;global.workoutSourceLabel=()=>'';global.prInfo=()=>[];global.allExercises=()=>[];global.toast=()=>{};global.esc=x=>String(x??'');global.fmt=x=>String(x);global.showModal=()=>{};global.closeModal=()=>{};global.setTimeout=()=>{};global.onNativeResult=()=>{};
global.dayData=date=>{let meals=state.meals.filter(x=>x.date===date),workouts=state.workouts.filter(x=>x.date===date),water=state.water.filter(x=>x.date===date),body=state.body.filter(x=>x.date===date);return {meals,workouts,water,body,calories:meals.reduce((a,x)=>a+num(x.calories),0),protein:meals.reduce((a,x)=>a+num(x.protein),0),carbs:meals.reduce((a,x)=>a+num(x.carbs),0),fat:meals.reduce((a,x)=>a+num(x.fat),0),fiber:meals.reduce((a,x)=>a+num(x.fiber),0),waterOz:water.reduce((a,x)=>a+num(x.oz),0),minutes:workouts.reduce((a,x)=>a+num(x.duration),0)}};
vm.runInThisContext(fs.readFileSync('fitness-android/app/src/main/assets/reportaddon.js','utf8'));
let html=reportCenterHtml();if(!html.includes('Needs update'))throw new Error('Legacy report was not marked stale');
generateAiReport('weekly','2026-08-19');
if(!lastReq.data.data_change_context)throw new Error('Missing data_change_context');
if(!lastReq.data.data_change_context.retroactive_updates.some(x=>x.date==='2026-08-18'&&x.category==='nutrition'))throw new Error('Backfilled prior-day meal was not detected');
onNativeResult(JSON.stringify({kind:'period_report',report_type:'weekly',start_date:'2026-08-17',end_date:'2026-08-23',label:'Weekly',text:'new report'}));
let saved=state.aiReports.find(x=>x.type==='weekly');if(!saved.sourceSignature)throw new Error('Report source signature was not saved');
html=reportCenterHtml();if(!html.includes('Current'))throw new Error('Fresh report did not become Current');
state.meals.push({id:'m2',date:'2026-08-18',name:'Snack',calories:100,protein:5,loggedAt:'2026-08-19T19:00:00Z'});
html=reportCenterHtml();if(!html.includes('Needs update'))throw new Error('Changed data did not stale the saved report');
console.log('v1.1.6 report freshness test PASS');

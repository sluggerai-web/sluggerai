/* Fitness Command Center v1.1.3 add-on: Gemini Daily / Weekly / Monthly Report Center */
(function(){
  state.version=Math.max(Number(state.version)||5,6);
  state.aiReports=Array.isArray(state.aiReports)?state.aiReports:[];
  state.reportSettings=Object.assign({
    dailyEnabled:false,
    dailyHour:20,
    weeklyEnabled:false,
    weeklyDay:0,
    weeklyHour:20,
    monthlyEnabled:false
  },state.reportSettings||{});
  save();

  const v113OldNativeResult=window.onNativeResult;
  const v113OldRenderToday=renderToday;
  const v113OldRenderProgress=renderProgress;
  const v113OldRenderSettings=renderSettings;
  window.pendingGeminiReport=null;
  window._autoReportBusy=false;

  function isoDate(d){return d.toISOString().slice(0,10)}
  function parseDate(s){return new Date((s||today())+'T12:00:00')}
  function addDays(s,n){let d=parseDate(s);d.setDate(d.getDate()+n);return isoDate(d)}
  function daysBetween(start,end){let out=[],d=parseDate(start),e=parseDate(end);while(d<=e){out.push(isoDate(d));d.setDate(d.getDate()+1)}return out}
  function monthName(date){return parseDate(date).toLocaleDateString(undefined,{month:'long',year:'numeric'})}

  window.reportPeriod=function(kind,anchor=today()){
    let a=parseDate(anchor),start,end,label;
    if(kind==='daily'){
      start=end=isoDate(a);label='Daily Report — '+dateLabel(start);
    }else if(kind==='weekly'){
      let dow=a.getDay(),back=(dow+6)%7,s=new Date(a);s.setDate(a.getDate()-back);let e=new Date(s);e.setDate(s.getDate()+6);
      start=isoDate(s);end=isoDate(e);label='Weekly Report — '+dateLabel(start)+' to '+dateLabel(end);
    }else{
      let s=new Date(a.getFullYear(),a.getMonth(),1,12),e=new Date(a.getFullYear(),a.getMonth()+1,0,12);
      start=isoDate(s);end=isoDate(e);label='Monthly Report — '+monthName(start);
    }
    return {kind,start,end,label,anchor};
  };

  function previousPeriod(p){
    if(p.kind==='daily'){let d=addDays(p.start,-1);return {kind:'daily',start:d,end:d,label:'Previous day'}}
    if(p.kind==='weekly'){return {kind:'weekly',start:addDays(p.start,-7),end:addDays(p.end,-7),label:'Previous week'}}
    let d=parseDate(p.start),prev=new Date(d.getFullYear(),d.getMonth()-1,1,12),prevEnd=new Date(d.getFullYear(),d.getMonth(),0,12);
    return {kind:'monthly',start:isoDate(prev),end:isoDate(prevEnd),label:'Previous month'};
  }

  function cleanMeal(m){return {name:m.name||'',type:m.type||'',calories:num(m.calories),protein_g:num(m.protein),carbs_g:num(m.carbs),fat_g:num(m.fat),fiber_g:num(m.fiber),source:m.source||''}}
  function cleanWorkout(x){return {exercise:x.exercise||'',workout_type:x.workoutType||x.plan||'',source:workoutSourceLabel(x),sets:num(x.sets),reps:num(x.reps),weight_lb:num(x.load),volume_lb_reps:num(x.volume),duration_min:num(x.duration),cardio_min:num(x.cardio),calories_burned_kcal:num(x.caloriesBurned),distance:num(x.distance),time_seconds:num(x.time),metric_name:x.metricName||'',metric_value:num(x.metricValue),unit:x.unit||'',rpe:num(x.rpe),notes:x.notes||''}}
  function daySnapshot(date){
    let d=dayData(date),h=healthFor(date),body=(d.body||[])[0]||{},habit=state.habits&&state.habits[date]?state.habits[date]:{};
    return {
      date,
      nutrition:{calories:num(d.calories),protein_g:num(d.protein),carbs_g:num(d.carbs),fat_g:num(d.fat),fiber_g:num(d.fiber),meals:(d.meals||[]).slice(0,12).map(cleanMeal)},
      hydration_oz:num(d.waterOz),
      workout_summary:{sessions:workoutSessions(d.workouts||[]),exercise_entries:(d.workouts||[]).length,minutes:num(d.minutes),strength_volume_lb_reps:(d.workouts||[]).reduce((a,x)=>a+num(x.volume),0),workout_calories_logged:(d.workouts||[]).reduce((a,x)=>a+num(x.caloriesBurned),0)},
      exercises:(d.workouts||[]).slice(0,24).map(cleanWorkout),
      body_recovery:{weight_lb:num(body.weight),sleep_hours:num(body.sleep),energy_score:num(body.energy),manual_resting_hr_bpm:num(body.hr)},
      habit_completion_pct:Math.round(habitPct(date)*100),
      health_connect:{average_hr_bpm:num(h.average_heart_rate_bpm),resting_hr_bpm:num(h.resting_hr_bpm),min_hr_bpm:num(h.min_heart_rate_bpm),max_hr_bpm:num(h.max_heart_rate_bpm),latest_hr_bpm:num(h.latest_heart_rate_bpm),steps:num(h.steps),active_calories_kcal:num(h.active_calories_kcal),total_calories_kcal:num(h.total_calories_kcal),sleep_hours:num(h.sleep_hours),distance_miles:num(h.distance_miles),weight_lb:num(h.weight_lb),exercise_sessions:num(h.exercise_sessions)}
    };
  }

  function periodSummary(start,end){
    let dates=daysBetween(start,end),snaps=dates.map(daySnapshot),n=dates.length||1;
    let sum=(fn)=>snaps.reduce((a,x)=>a+num(fn(x)),0),vals=(fn)=>snaps.map(fn).map(num).filter(v=>v>0),avg=(fn)=>{let a=vals(fn);return a.length?a.reduce((x,y)=>x+y,0)/a.length:0};
    let weights=snaps.map(x=>num(x.body_recovery.weight_lb)||num(x.health_connect.weight_lb)).filter(v=>v>0);
    let prs=[];dates.forEach(date=>(state.workouts||[]).filter(x=>x.date===date&&x.completed!==false).forEach(x=>prInfo(x).forEach(p=>prs.push({date,exercise:x.exercise,achievement:p}))));
    return {
      days:n,
      days_with_meals:snaps.filter(x=>x.nutrition.meals.length).length,
      meals_logged:sum(x=>x.nutrition.meals.length),
      average_daily_calories:sum(x=>x.nutrition.calories)/n,
      average_daily_protein_g:sum(x=>x.nutrition.protein_g)/n,
      average_daily_carbs_g:sum(x=>x.nutrition.carbs_g)/n,
      average_daily_fat_g:sum(x=>x.nutrition.fat_g)/n,
      average_daily_fiber_g:sum(x=>x.nutrition.fiber_g)/n,
      average_daily_water_oz:sum(x=>x.hydration_oz)/n,
      workout_sessions:sum(x=>x.workout_summary.sessions),
      exercise_entries:sum(x=>x.workout_summary.exercise_entries),
      workout_minutes:sum(x=>x.workout_summary.minutes),
      strength_volume_lb_reps:sum(x=>x.workout_summary.strength_volume_lb_reps),
      workout_calories_logged:sum(x=>x.workout_summary.workout_calories_logged),
      average_habit_completion_pct:sum(x=>x.habit_completion_pct)/n,
      weight_change_lb:weights.length>1?weights[weights.length-1]-weights[0]:0,
      health_days_synced:snaps.filter(x=>num(x.health_connect.average_hr_bpm)||num(x.health_connect.steps)||num(x.health_connect.active_calories_kcal)).length,
      average_heart_rate_bpm:avg(x=>x.health_connect.average_hr_bpm),
      average_resting_hr_bpm:avg(x=>x.health_connect.resting_hr_bpm),
      average_steps:avg(x=>x.health_connect.steps),
      average_active_calories_kcal:avg(x=>x.health_connect.active_calories_kcal),
      average_total_calories_kcal:avg(x=>x.health_connect.total_calories_kcal),
      average_sleep_hours:avg(x=>x.health_connect.sleep_hours||x.body_recovery.sleep_hours),
      average_distance_miles:avg(x=>x.health_connect.distance_miles),
      personal_bests:prs.slice(0,30)
    };
  }

  window.buildReportPayload=function(kind,anchor=today()){
    let p=reportPeriod(kind,anchor),prev=previousPeriod(p),dates=daysBetween(p.start,p.end),detail=dates.map(daySnapshot);
    return {
      report_type:p.kind,
      label:p.label,
      start_date:p.start,
      end_date:p.end,
      generated_for_date:today(),
      goals:state.goals,
      current_period_summary:periodSummary(p.start,p.end),
      previous_period_summary:periodSummary(prev.start,prev.end),
      daily_detail:detail,
      notes:{health_data_source:'Health Connect / Pixel Watch when synced',meal_photo_values:'Gemini photo values are estimates',missing_values:'Zero values may mean not logged or unavailable; do not invent missing data.'}
    };
  };

  function latestReport(type,start,end){return (state.aiReports||[]).filter(r=>(!type||r.type===type)&&(!start||r.start===start)&&(!end||r.end===end)).sort((a,b)=>String(b.generatedAt).localeCompare(String(a.generatedAt)))[0]||null}
  function reportExists(type,start,end){return !!latestReport(type,start,end)}

  window.generateAiReport=function(kind,anchor=today(),auto=false){
    if(!native()||!Android.generateReport)return toast('This app build does not have the Gemini report bridge.');
    if(!Android.hasApiKey||!Android.hasApiKey())return toast('Add your Gemini API key in Settings first.');
    if(window.pendingGeminiReport)return toast('A Gemini report is already being generated.');
    let p=reportPeriod(kind,anchor),payload=buildReportPayload(kind,anchor);
    window.pendingGeminiReport={type:p.kind,start:p.start,end:p.end,label:p.label,auto:!!auto,requestId:uid('RPT')};
    let status=$('#reportStatus')||$('#dailyReportStatus');if(status)status.textContent='Gemini is reviewing meals, exercise, hydration, recovery and Health Connect data…';
    Android.generateReport(JSON.stringify({request_id:window.pendingGeminiReport.requestId,report_type:p.kind,label:p.label,start_date:p.start,end_date:p.end,auto:!!auto,data:payload}));
  };

  window.deleteAiReport=function(id){state.aiReports=(state.aiReports||[]).filter(r=>r.id!==id);save();render();toast('Report deleted')};
  window.showAiReport=function(id){let r=(state.aiReports||[]).find(x=>x.id===id);if(!r)return;showModal(`<div class="between"><div><div class="eyebrow">Gemini ${esc(r.type)} report</div><div class="section-title">${esc(r.label)}</div></div><span class="pill">AI</span></div><div class="small muted">Generated ${new Date(r.generatedAt).toLocaleString()}</div><div class="card mt"><div class="small" style="white-space:pre-wrap;line-height:1.55">${esc(r.text||'')}</div></div><div class="row mt"><button class="btn light" onclick="closeModal()">Close</button><button class="btn danger" onclick="deleteAiReport('${r.id}');closeModal()">Delete</button></div>`)};

  window.reportCardHtml=function(r,compact=false){if(!r)return '<div class="small muted">No Gemini report generated for this period yet.</div>';let preview=String(r.text||'').replace(/\s+/g,' ').trim();let lim=compact?260:420;if(preview.length>lim)preview=preview.slice(0,lim)+'…';return `<div class="card mt"><div class="between"><div><b>${esc(r.label)}</b><div class="tiny muted">Generated ${new Date(r.generatedAt).toLocaleString()}${r.auto?' • automatic':''}</div></div><span class="pill">Gemini</span></div><div class="small mt" style="white-space:pre-wrap">${esc(preview)}</div><button class="btn sm light mt" onclick="showAiReport('${r.id}')">Open full report</button></div>`};

  window.reportCenterHtml=function(){
    let anchor=today(),daily=reportPeriod('daily',anchor),week=reportPeriod('weekly',anchor),month=reportPeriod('monthly',anchor),recent=(state.aiReports||[]).slice().sort((a,b)=>String(b.generatedAt).localeCompare(String(a.generatedAt))).slice(0,8);
    return `<div class="card mt"><div class="between"><div><b>Gemini Report Center</b><div class="tiny muted">Daily, weekly and monthly coaching summaries from your actual logged data.</div></div><span class="pill">AI reports</span></div>
      <label class="mt"><span>Report anchor date</span><input id="reportAnchor" type="date" value="${anchor}"></label>
      <div class="row wrap"><button class="btn teal" onclick="generateAiReport('daily',$('#reportAnchor').value)">Daily report</button><button class="btn gold" onclick="generateAiReport('weekly',$('#reportAnchor').value)">Weekly report</button><button class="btn light" onclick="generateAiReport('monthly',$('#reportAnchor').value)">Monthly report</button></div>
      <div id="reportStatus" class="small muted mt">Gemini reviews nutrition, workouts, water, body/recovery, habits, heart rate, steps, sleep and calories burned when those values are available.</div>
      <div class="grid3 mt"><div><div class="tiny muted">Today</div><b>${latestReport('daily',daily.start,daily.end)?'Ready':'—'}</b></div><div><div class="tiny muted">This week</div><b>${latestReport('weekly',week.start,week.end)?'Ready':'—'}</b></div><div><div class="tiny muted">This month</div><b>${latestReport('monthly',month.start,month.end)?'Ready':'—'}</b></div></div>
    </div><div class="section-title">Gemini report history</div><div>${recent.length?recent.map(r=>reportCardHtml(r,true)).join(''):'<div class="card"><div class="small muted">No saved Gemini reports yet. Generate a Daily, Weekly or Monthly report above.</div></div>'}</div>`;
  };

  function settingChecked(k){return state.reportSettings[k]?'checked':''}
  window.reportSettingsHtml=function(){let s=state.reportSettings;return `<div class="card mt"><div class="between"><div><b>Automatic Gemini Reports</b><div class="tiny muted">Optional. Reports generate the next time the app is open after the scheduled time.</div></div><span class="pill">Opt-in</span></div>
    <label class="check mt"><input id="rsDaily" type="checkbox" ${settingChecked('dailyEnabled')}><span><b>Daily report</b><small>Generate after the selected evening hour.</small></span></label>
    <label><span>Daily report hour</span><select id="rsDailyHour"><option value="18" ${s.dailyHour==18?'selected':''}>6:00 PM</option><option value="19" ${s.dailyHour==19?'selected':''}>7:00 PM</option><option value="20" ${s.dailyHour==20?'selected':''}>8:00 PM</option><option value="21" ${s.dailyHour==21?'selected':''}>9:00 PM</option><option value="22" ${s.dailyHour==22?'selected':''}>10:00 PM</option></select></label>
    <label class="check mt"><input id="rsWeekly" type="checkbox" ${settingChecked('weeklyEnabled')}><span><b>Weekly report</b><small>Generate on the selected day after the selected hour.</small></span></label>
    <div class="form2"><label><span>Weekly report day</span><select id="rsWeeklyDay"><option value="0" ${s.weeklyDay==0?'selected':''}>Sunday</option><option value="6" ${s.weeklyDay==6?'selected':''}>Saturday</option><option value="5" ${s.weeklyDay==5?'selected':''}>Friday</option></select></label><label><span>Weekly report hour</span><select id="rsWeeklyHour"><option value="18" ${s.weeklyHour==18?'selected':''}>6:00 PM</option><option value="19" ${s.weeklyHour==19?'selected':''}>7:00 PM</option><option value="20" ${s.weeklyHour==20?'selected':''}>8:00 PM</option><option value="21" ${s.weeklyHour==21?'selected':''}>9:00 PM</option></select></label></div>
    <label class="check mt"><input id="rsMonthly" type="checkbox" ${settingChecked('monthlyEnabled')}><span><b>Monthly report</b><small>Generate the prior calendar month the first time you open the app in a new month.</small></span></label>
    <button class="btn block teal mt" onclick="saveReportSettings()">Save report schedule</button>
    <p class="tiny muted mt">This uses your Gemini API key and internet connection. It does not run a permanent background service; if the app is closed at the scheduled time, the report is generated on the next app open when due.</p></div>`};

  window.saveReportSettings=function(){state.reportSettings.dailyEnabled=$('#rsDaily').checked;state.reportSettings.dailyHour=Number($('#rsDailyHour').value)||20;state.reportSettings.weeklyEnabled=$('#rsWeekly').checked;state.reportSettings.weeklyDay=Number($('#rsWeeklyDay').value);state.reportSettings.weeklyHour=Number($('#rsWeeklyHour').value)||20;state.reportSettings.monthlyEnabled=$('#rsMonthly').checked;save();toast('Gemini report schedule saved');setTimeout(checkAutoReports,500)};

  window.dailyReportHtml=function(date){let p=reportPeriod('daily',date),r=latestReport('daily',p.start,p.end);return `<div class="card mt"><div class="between"><div><b>Gemini Daily Report</b><div class="tiny muted">Meals • exercise • water • HR/recovery • what went well • what to improve</div></div><span class="pill">AI</span></div><button class="btn block gold mt" onclick="generateAiReport('daily','${date}')">${r?'Regenerate':'Generate'} report for this day</button><div id="dailyReportStatus" class="small muted mt"></div>${r?reportCardHtml(r,true):''}</div>`};

  window.renderToday=function(){v113OldRenderToday();let c=$('#content');if(c)c.insertAdjacentHTML('beforeend',dailyReportHtml(window.dailyActivityDate||today()))};
  window.renderProgress=function(){v113OldRenderProgress();let c=$('#content');if(c)c.insertAdjacentHTML('afterbegin',reportCenterHtml())};
  window.renderSettings=function(){v113OldRenderSettings();let c=$('#content');if(c)c.insertAdjacentHTML('beforeend',reportSettingsHtml()+`<div class="card mt"><b>About v1.1.3 reports</b><p class="small">Gemini Daily, Weekly and Monthly reports are saved locally in the app and included in backup exports. They use only the data you have logged/synced and are designed for coaching-style feedback, not medical diagnosis.</p></div>`) };

  window.checkAutoReports=function(){
    if(window.pendingGeminiReport||window._autoReportBusy||!native()||!Android.generateReport||!Android.hasApiKey||!Android.hasApiKey())return;
    let s=state.reportSettings||{},now=new Date(),hour=now.getHours(),anchor=today();
    if(s.dailyEnabled&&hour>=Number(s.dailyHour||20)){
      let p=reportPeriod('daily',anchor);if(!reportExists('daily',p.start,p.end)){window._autoReportBusy=true;generateAiReport('daily',anchor,true);return}
    }
    if(s.weeklyEnabled&&now.getDay()===Number(s.weeklyDay??0)&&hour>=Number(s.weeklyHour||20)){
      let p=reportPeriod('weekly',anchor);if(!reportExists('weekly',p.start,p.end)){window._autoReportBusy=true;generateAiReport('weekly',anchor,true);return}
    }
    if(s.monthlyEnabled){
      let prev=new Date(now.getFullYear(),now.getMonth()-1,15,12),a=isoDate(prev),p=reportPeriod('monthly',a);
      if(!reportExists('monthly',p.start,p.end)){window._autoReportBusy=true;generateAiReport('monthly',a,true);return}
    }
  };

  window.onNativeResult=function(raw){
    let r;try{r=typeof raw==='string'?JSON.parse(raw):raw}catch(e){return v113OldNativeResult(raw)}
    if(r.kind==='period_report'){
      let p=window.pendingGeminiReport||{type:r.report_type||'report',start:r.start_date||'',end:r.end_date||'',label:r.label||'Gemini Report',auto:!!r.auto,requestId:r.request_id||uid('RPT')};
      let entry={id:uid('AIR'),type:r.report_type||p.type,start:r.start_date||p.start,end:r.end_date||p.end,label:r.label||p.label,generatedAt:new Date().toISOString(),auto:typeof r.auto==='boolean'?r.auto:p.auto,text:r.text||''};
      state.aiReports=(state.aiReports||[]).filter(x=>!(x.type===entry.type&&x.start===entry.start&&x.end===entry.end));state.aiReports.push(entry);save();window.pendingGeminiReport=null;window._autoReportBusy=false;toast('Gemini '+entry.type+' report saved');render();setTimeout(checkAutoReports,1200);return;
    }
    if(r.kind==='report_error'){window.pendingGeminiReport=null;window._autoReportBusy=false;let status=$('#reportStatus')||$('#dailyReportStatus');if(status)status.textContent=r.message||'Report generation failed';toast(r.message||'Gemini report failed');return}
    let result=v113OldNativeResult(raw);
    if(r.kind==='health_data')setTimeout(checkAutoReports,900);
    return result;
  };

  setTimeout(checkAutoReports,1800);
  render();
})();

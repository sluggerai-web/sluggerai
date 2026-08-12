/* Fitness Command Center v1.1.2 add-on: Health Connect + Gemini Voice + Daily Activity review */
(function(){
  state.version=Math.max(Number(state.version)||3,5);
  state.healthHistory=state.healthHistory||{};
  state.healthSyncEnabled=state.healthSyncEnabled!==false;
  save();

  const oldNativeResult=window.onNativeResult;
  const oldRenderHome=renderHome;
  const oldRenderTraining=renderTraining;
  const oldRenderFood=renderFood;
  const oldRenderToday=renderToday;
  const oldRenderProgress=renderProgress;
  const oldRenderSettings=renderSettings;
  const oldWeeklyReport=weeklyReport;
  const oldCurrentWeight=currentWeight;

  window.dailyActivityDate=window.dailyActivityDate||today();
  window.pendingHealthDate=null;

  window.healthFor=function(date=today()){return state.healthHistory&&state.healthHistory[date]?state.healthHistory[date]:{}};
  window.currentWeight=function(){let x=oldCurrentWeight();if(x)return x;let dates=Object.keys(state.healthHistory||{}).sort();for(let i=dates.length-1;i>=0;i--){let w=num((state.healthHistory[dates[i]]||{}).weight_lb);if(w)return w}return 0};
  window.dateLabel=function(date){try{return new Date(date+'T12:00:00').toLocaleDateString(undefined,{weekday:'short',month:'short',day:'numeric',year:'numeric'})}catch(e){return date}};

  window.healthCardHtml=function(compact=false,date=today()){
    let h=healthFor(date), available=native()&&Android.healthConnectAvailable?Android.healthConnectAvailable():false;
    let has=Object.keys(h||{}).length>0;
    let source=esc(h.source_label||'Health Connect / Pixel Watch');
    let todayFlag=date===today();
    return `<div class="card mt health-card"><div class="between"><div><b>Health Connect</b><div class="tiny muted">${source}${todayFlag?'':' • '+esc(dateLabel(date))}</div></div><span class="pill ${has?'green':available?'':'warn'}">${has?'Synced':available?'Ready':'Unavailable'}</span></div>
      <div class="health-grid mt">
        <div class="health-stat"><span>Average HR</span><b>${num(h.average_heart_rate_bpm)?fmt(h.average_heart_rate_bpm)+' bpm':'—'}</b></div>
        <div class="health-stat"><span>Latest HR</span><b>${num(h.latest_heart_rate_bpm)?fmt(h.latest_heart_rate_bpm)+' bpm':'—'}</b></div>
        <div class="health-stat"><span>Resting HR</span><b>${num(h.resting_hr_bpm)?fmt(h.resting_hr_bpm)+' bpm':'—'}</b></div>
        <div class="health-stat"><span>Steps</span><b>${num(h.steps)?fmt(h.steps):'—'}</b></div>
        ${compact?'':`<div class="health-stat"><span>Active kcal burned</span><b>${num(h.active_calories_kcal)?fmt(h.active_calories_kcal):'—'}</b></div><div class="health-stat"><span>Total kcal burned</span><b>${num(h.total_calories_kcal)?fmt(h.total_calories_kcal):'—'}</b></div><div class="health-stat"><span>Sleep</span><b>${num(h.sleep_hours)?fmt(h.sleep_hours,1)+' hr':'—'}</b></div><div class="health-stat"><span>Distance</span><b>${num(h.distance_miles)?fmt(h.distance_miles,2)+' mi':'—'}</b></div>`}
      </div>
      <div class="row wrap mt"><button class="btn sm teal" onclick="syncHealthDate('${date}')">Sync ${todayFlag?'today':'this day'}</button><button class="btn sm light" onclick="connectHealth()">Manage access</button>${todayFlag&&has?'<button class="btn sm light" onclick="useHealthRecovery()">Use recovery values</button>':''}</div>
      <div id="healthStatus" class="tiny muted mt">${has&&h.synced_at?'Last synced '+new Date(h.synced_at).toLocaleTimeString([], {hour:'numeric',minute:'2-digit'}):'Health and watch data are read only after you grant Health Connect permission.'}</div></div>`;
  };

  window.voiceCardHtml=function(){return `<div class="card mt voice-card"><div class="between"><div><b>Voice Quick Log</b><div class="tiny muted">Speak naturally — Gemini turns it into a reviewable log.</div></div><span class="pill">🎙 Gemini</span></div><p class="small muted">Examples: “Bench press, 3 sets of 8 at 135 pounds.” “Lunch was two eggs, toast and a banana.” “I drank 20 ounces of water.” If you know workout calories, you can say “I burned 280 calories.” You review everything before it is saved.</p><button class="btn block teal" onclick="startVoiceLog('auto')">🎙 Speak food, workout, water or recovery</button><div id="voiceStatus" class="small muted mt"></div></div>`};

  window.workoutSourceLabel=function(x){if(x.source)return x.source;if(x.plan==='Voice Log')return 'Gemini Voice';if(x.plan==='Manual'||x.workoutType==='Manual')return 'Manual';if(x.workoutType==='Performance Test')return 'Performance Test';if(x.plan)return 'Plan';return 'Workout log'};
  window.sessionGroupsFor=function(date){let rows=(state.workouts||[]).filter(x=>x.date===date&&x.completed!==false);let map=new Map();rows.forEach((x,i)=>{let k=x.sessionId||x.id||('row-'+i);if(!map.has(k))map.set(k,[]);map.get(k).push(x)});return Array.from(map.values())};
  window.sessionBurn=function(rows){return (rows||[]).reduce((a,x)=>a+num(x.caloriesBurned),0)};
  window.sessionDuration=function(rows){return (rows||[]).reduce((a,x)=>a+num(x.duration),0)};
  window.exerciseDetailText=function(x){let a=[];if(num(x.sets))a.push(fmt(x.sets)+' sets');if(num(x.reps))a.push('× '+fmt(x.reps)+' reps');if(num(x.load))a.push('@ '+fmt(x.load)+' lb');if(num(x.duration))a.push(fmt(x.duration)+' min');if(num(x.cardio))a.push(fmt(x.cardio)+' cardio min');if(num(x.distance))a.push(fmt(x.distance,2)+(String(x.unit||'').toLowerCase().includes('mile')?' mi':' yd'));if(num(x.time))a.push(fmt(x.time,2)+' sec');if(num(x.metricValue))a.push(esc(x.metricName||'Result')+' '+fmt(x.metricValue,2)+' '+esc(x.unit||''));if(num(x.rpe))a.push('RPE '+fmt(x.rpe));return a.length?a.join(' • '):'Completed'};

  window.prInfo=function(x){
    let prior=(state.workouts||[]).filter(q=>q.exercise===x.exercise&&q.date<x.date&&q.completed!==false);
    let out=[];
    if(num(x.load)>0){let best=Math.max(0,...prior.map(q=>num(q.load)));if(best>0&&num(x.load)>best)out.push(`Load PR +${fmt(num(x.load)-best,1)} lb`)}
    if(num(x.metricValue)>0){let vals=prior.map(q=>num(q.metricValue)).filter(v=>v>0);if(vals.length){let lib=allExercises().find(q=>q.name===x.exercise)||{},lower=String(lib.better||'Higher').toLowerCase()==='lower';let priorBest=lower?Math.min(...vals):Math.max(...vals);let better=lower?num(x.metricValue)<priorBest:num(x.metricValue)>priorBest;if(better){let delta=Math.abs(num(x.metricValue)-priorBest);out.push(`${esc(x.metricName||'Result')} PR ${lower?'-':'+'}${fmt(delta,2)} ${esc(x.unit||'')}`)}}}
    if(num(x.volume)>0){let vols=prior.map(q=>num(q.volume)).filter(v=>v>0);let best=vols.length?Math.max(...vols):0;if(best>0&&num(x.volume)>best)out.push(`Volume PR +${fmt(num(x.volume)-best)} lb-reps`)}
    return out;
  };

  window.localWorkoutHtml=function(date){
    let groups=sessionGroupsFor(date);if(!groups.length)return '<div class="small muted">No exercises were logged for this day.</div>';
    return groups.map((rows,gi)=>{let first=rows[0]||{},burn=sessionBurn(rows),dur=sessionDuration(rows),src=workoutSourceLabel(first);return `<div class="exercise"><div class="between"><div><div class="name">${esc(first.workoutType||first.plan||'Workout session')}</div><div class="tiny muted">${esc(src)} • ${rows.length} exercise${rows.length===1?'':'s'}${dur?' • '+fmt(dur)+' min':''}</div></div>${burn?`<span class="pill green">${fmt(burn)} kcal burned</span>`:'<span class="pill">Logged</span>'}</div>${rows.map(x=>{let prs=prInfo(x);return `<div class="statline"><div><b>${esc(x.exercise||'Exercise')}</b><div class="tiny muted">${exerciseDetailText(x)}</div>${x.notes?`<div class="tiny muted">${esc(x.notes)}</div>`:''}</div><div class="row wrap">${prs.length?`<span class="pill green">PR</span>`:''}</div></div>`}).join('')}</div>`}).join('');
  };

  window.healthExerciseHtml=function(date){let h=healthFor(date),a=Array.isArray(h.exercise_sessions_detail)?h.exercise_sessions_detail:[];if(!a.length)return '<div class="small muted">No Health Connect exercise sessions are synced for this day.</div>';return a.map(x=>{let title=x.title||'Health / watch workout';return `<div class="statline"><div><b>${esc(title)}</b><div class="tiny muted">${num(x.duration_min)?fmt(x.duration_min)+' min':''}${num(x.average_heart_rate_bpm)?' • avg HR '+fmt(x.average_heart_rate_bpm)+' bpm':''}${x.source_package?' • '+esc(x.source_package):''}</div></div><span class="pill ${num(x.active_calories_kcal)?'green':''}">${num(x.active_calories_kcal)?fmt(x.active_calories_kcal)+' active kcal':'Synced'}</span></div>`}).join('')};

  window.dailyProgressHtml=function(date){let rows=(state.workouts||[]).filter(x=>x.date===date&&x.completed!==false),prs=[];rows.forEach(x=>prInfo(x).forEach(p=>prs.push({exercise:x.exercise,text:p})));let volume=rows.reduce((a,x)=>a+num(x.volume),0),metrics=rows.filter(x=>num(x.metricValue)>0).length;return `<div class="grid3"><div><div class="tiny muted">Exercises</div><b>${fmt(rows.length)}</b></div><div><div class="tiny muted">Strength volume</div><b>${fmt(volume)} lb-reps</b></div><div><div class="tiny muted">New PRs</div><b>${fmt(prs.length)}</b></div></div>${prs.length?`<div class="mt">${prs.map(p=>`<div class="statline"><span>${esc(p.exercise)}</span><span class="pill green">${p.text}</span></div>`).join('')}</div>`:`<p class="small muted mt">${rows.length?'No new personal best was detected from the comparable prior entries. The completed work is still included in your daily and long-term reports.':'Log a workout to start tracking day-by-day performance changes.'}</p>`}${metrics?`<div class="tiny muted mt">${metrics} performance result${metrics===1?'':'s'} recorded for this day.</div>`:''}`};

  window.dailyActivityHtml=function(date=window.dailyActivityDate||today()){
    let d=dayData(date),h=healthFor(date),groups=sessionGroupsFor(date),localBurn=groups.reduce((a,g)=>a+sessionBurn(g),0),vol=d.workouts.reduce((a,x)=>a+num(x.volume),0),sessions=groups.length;
    let active=num(h.active_calories_kcal),total=num(h.total_calories_kcal),avgHr=num(h.average_heart_rate_bpm),isToday=date===today();
    return `<div class="card mb"><div class="between"><div><div class="eyebrow">Daily activity & progress</div><div class="section-title" style="margin:3px 0">${isToday?'Today':esc(dateLabel(date))}</div></div><span class="pill ${sessions?'green':''}">${sessions?fmt(sessions)+' workout'+(sessions===1?'':'s'):'No workout yet'}</span></div>
      <div class="row wrap mt"><button class="btn sm light" onclick="shiftDailyActivityDate(-1)">← Day</button><input id="dailyActivityDateInput" type="date" value="${date}" max="${today()}" style="width:auto;padding:7px 8px" onchange="setDailyActivityDate(this.value)"><button class="btn sm light" onclick="shiftDailyActivityDate(1)" ${isToday?'disabled':''}>Day →</button><button class="btn sm teal" onclick="setDailyActivityDate('${today()}')">Today</button></div>
      <div class="grid mt">${cardKpi('Exercises completed',fmt(d.workouts.length),`${fmt(d.minutes)} workout min`)}${cardKpi('Strength volume',fmt(vol),'lb-reps')}${cardKpi('Calories eaten',fmt(d.calories),`${fmt(d.protein)}g protein`)}${cardKpi('Active kcal burned',active?fmt(active):localBurn?fmt(localBurn):'—',active?'Health Connect':localBurn?'logged workout value':'sync Health Connect')}${cardKpi('Total kcal burned',total?fmt(total):'—',total?'Health Connect incl. resting burn':'sync Health Connect')}${cardKpi('Average heart rate',avgHr?fmt(avgHr)+' bpm':'—',avgHr?`${fmt(h.min_heart_rate_bpm)}–${fmt(h.max_heart_rate_bpm)} bpm range`:'sync Health Connect')}${cardKpi('Water',fmt(d.waterOz)+' oz',`${Math.round(pct(d.waterOz,state.goals.water)*100)}% of goal`)}${cardKpi('Workout calories logged',localBurn?fmt(localBurn):'—','manual/voice value if supplied')}</div>
      <div class="row wrap mt"><button class="btn sm teal" onclick="syncHealthDate('${date}')">Sync watch/Health data for this day</button>${!isToday?'<button class="btn sm light" onclick="setPage(\'progress\')">Back to Progress</button>':''}</div></div>
      <div class="card mb"><div class="between"><b>Exercises completed</b><span class="pill">${fmt(d.workouts.length)} entries</span></div><div class="mt">${localWorkoutHtml(date)}</div></div>
      <div class="card mb"><div class="between"><b>Progress made this day</b><span class="pill">PR check</span></div><div class="mt">${dailyProgressHtml(date)}</div></div>
      <div class="card mb"><div class="between"><b>Pixel Watch / Health Connect workouts</b><span class="pill">${fmt(h.exercise_sessions||0)} sessions</span></div><p class="tiny muted">These are device-recorded sessions. Calories and average HR below come from the Health Connect time interval for that watch/health workout.</p><div>${healthExerciseHtml(date)}</div></div>`;
  };

  window.refreshDailyActivity=function(){let el=$('#dailyActivityPanel');if(el)el.innerHTML=dailyActivityHtml(window.dailyActivityDate)};
  window.setDailyActivityDate=function(date){if(!date||date>today())date=today();window.dailyActivityDate=date;refreshDailyActivity()};
  window.shiftDailyActivityDate=function(delta){let d=new Date((window.dailyActivityDate||today())+'T12:00:00');d.setDate(d.getDate()+delta);let v=d.toISOString().slice(0,10);if(v>today())v=today();setDailyActivityDate(v)};
  window.goToDay=function(date){window.dailyActivityDate=date;setPage('today')};
  window.dailyHistoryHtml=function(){let dates=new Set();(state.workouts||[]).forEach(x=>dates.add(x.date));Object.keys(state.healthHistory||{}).forEach(x=>dates.add(x));return Array.from(dates).filter(Boolean).sort().reverse().slice(0,10).map(d=>`<button class="btn sm light" onclick="goToDay('${d}')">${esc(dateLabel(d))}</button>`).join('')||'<span class="small muted">No daily history yet.</span>'};

  window.renderHome=function(){oldRenderHome();let c=$('#content');if(c){c.insertAdjacentHTML('beforeend',healthCardHtml(true)+voiceCardHtml())}};
  window.renderTraining=function(){oldRenderTraining();let c=$('#content');if(c&&!activePlan)c.insertAdjacentHTML('beforeend',`<div class="card mt"><b>Hands-free training log</b><p class="small muted">Say the exercise, sets, reps, weight, distance, sprint time, RPE, or known calories burned. Gemini will structure it, then ask you to confirm.</p><button class="btn block teal" onclick="startVoiceLog('workout')">🎙 Log workout by voice</button></div>`)};
  window.renderFood=function(){oldRenderFood();let c=$('#content');if(c)c.insertAdjacentHTML('afterbegin',`<div class="card mb"><div class="between"><b>Speak a meal</b><span class="pill">🎙</span></div><p class="small muted">Tell the app what you ate and any portions you know. Gemini estimates missing nutrition values, then shows them for approval.</p><button class="btn block teal" onclick="startVoiceLog('food')">🎙 Log food by voice</button></div>`)};
  window.renderToday=function(){oldRenderToday();let c=$('#content');if(c){c.insertAdjacentHTML('afterbegin',healthCardHtml(false));c.insertAdjacentHTML('afterbegin',`<div id="dailyActivityPanel">${dailyActivityHtml(window.dailyActivityDate)}</div>`)}};
  window.renderProgress=function(){try{oldRenderProgress()}catch(e){console.error('Base progress render failed',e);let c=$('#content');if(c)c.innerHTML='<div class="section-title">Progress</div><div class="notice error">Progress screen recovered from a display error. Your saved data is still intact.</div>'}let c=$('#content');if(c)c.insertAdjacentHTML('beforeend',`<div class="card mt"><div class="between"><b>Daily history</b><span class="pill">Tap a day</span></div><div class="row wrap mt">${dailyHistoryHtml()}</div></div>${healthProgressHtml()}`)};
  window.renderSettings=function(){oldRenderSettings();let c=$('#content');if(c)c.insertAdjacentHTML('beforeend',`<div class="card mt"><b>Pixel Watch / Health Connect</b><p class="small muted">Read average/latest/resting heart rate, steps, sleep, active and total calories burned, distance, weight and exercise sessions available in Health Connect.</p><div class="row wrap"><button class="btn teal" onclick="connectHealth()">Connect / permissions</button><button class="btn light" onclick="syncHealth()">Sync today</button><button class="btn light" onclick="openHealthSettings()">Health Connect settings</button></div><div id="healthStatus" class="small muted mt"></div></div>${voiceCardHtml()}<div class="card mt"><b>About this build</b><p class="small">Version 1.1.2 adds Daily Activity & Progress drill-downs, calories-burned visibility, Health Connect workout details, and average heart rate from Health Connect / Pixel Watch data.</p></div>`)};

  window.connectHealth=function(){if(!native()||!Android.requestHealthPermissions)return toast('Health Connect bridge unavailable');window.pendingHealthDate=window.pendingHealthDate||today();let el=$('#healthStatus');if(el)el.textContent='Opening Health Connect permissions…';Android.requestHealthPermissions()};
  window.syncHealth=function(){syncHealthDate(today())};
  window.syncHealthDate=function(date){if(!native())return toast('Health Connect bridge unavailable');window.pendingHealthDate=date||today();let el=$('#healthStatus');if(el)el.textContent='Reading Health Connect data for '+dateLabel(window.pendingHealthDate)+'…';if(Android.readHealthDate)Android.readHealthDate(window.pendingHealthDate);else if(window.pendingHealthDate===today()&&Android.readHealthToday)Android.readHealthToday();else toast('This app build cannot read historical health dates.')};
  window.openHealthSettings=function(){if(native()&&Android.openHealthConnectSettings)Android.openHealthConnectSettings()};
  window.useHealthRecovery=function(){let h=healthFor();if(!h||!Object.keys(h).length)return toast('Sync Health Connect first');let d=today(),existing=state.body.find(x=>x.date===d);let entry=existing||{id:uid('B'),date:d,weight:0,sleep:0,energy:0,hr:0};if(num(h.weight_lb))entry.weight=num(h.weight_lb);if(num(h.sleep_hours))entry.sleep=num(h.sleep_hours);if(num(h.resting_hr_bpm))entry.hr=num(h.resting_hr_bpm);if(!existing)state.body.push(entry);save();toast('Health recovery values added to today');render()};

  window.startVoiceLog=function(mode='auto'){if(!native()||!Android.startVoiceInput)return toast('Voice bridge unavailable');let el=$('#voiceStatus');if(el)el.textContent='Listening…';Android.startVoiceInput(mode)};
  window.parseVoiceTranscript=function(text,mode='auto'){if(!native()||!Android.parseVoiceLog)return;let el=$('#voiceStatus');if(el)el.textContent='Gemini is organizing: “'+text+'”';Android.parseVoiceLog(text,mode)};
  window.showVoiceReview=function(x,transcript){
    x=x||{};let meals=x.meals||[], ws=(x.workout_session||{}), exs=ws.exercises||[], water=num(x.water_oz), b=x.body||{};
    let details=[];
    meals.forEach(m=>details.push(`<div class="voice-item"><b>🍽 ${esc(m.food_name||'Meal')}</b><span>${fmt(m.calories)} kcal • P ${fmt(m.protein_g)}g • C ${fmt(m.carbs_g)}g • F ${fmt(m.fat_g)}g</span><small>${esc(m.meal_type||'')} ${m.serving_size?'• '+esc(m.serving_size):''}</small></div>`));
    exs.forEach(e=>details.push(`<div class="voice-item"><b>⚡ ${esc(e.exercise||'Exercise')}</b><span>${num(e.sets)?fmt(e.sets)+' sets ':''}${num(e.reps)?'× '+fmt(e.reps)+' reps ':''}${num(e.weight_lb)?'@ '+fmt(e.weight_lb)+' lb ':''}${num(e.time_seconds)?'• '+fmt(e.time_seconds,2)+' sec ':''}${num(e.distance_yards)?'• '+fmt(e.distance_yards)+' yd':''}</span><small>${num(e.rpe)?'RPE '+fmt(e.rpe):''}</small></div>`));
    if(num(ws.calories_burned_kcal))details.push(`<div class="voice-item"><b>🔥 Workout calories</b><span>${fmt(ws.calories_burned_kcal)} kcal burned</span><small>Saved because you explicitly stated this value.</small></div>`);
    if(water)details.push(`<div class="voice-item"><b>💧 Water</b><span>${fmt(water)} oz</span></div>`);
    if(num(b.weight_lb)||num(b.sleep_hours)||num(b.energy_score)||num(b.resting_hr_bpm))details.push(`<div class="voice-item"><b>♥ Recovery / body</b><span>${num(b.weight_lb)?fmt(b.weight_lb,1)+' lb ':''}${num(b.sleep_hours)?'• '+fmt(b.sleep_hours,1)+' hr sleep ':''}${num(b.energy_score)?'• energy '+fmt(b.energy_score):''}${num(b.resting_hr_bpm)?'• RHR '+fmt(b.resting_hr_bpm):''}</span></div>`);
    showModal(`<div class="between"><div><div class="eyebrow">Gemini voice log</div><div class="section-title">Review before saving</div></div><span class="pill">${Math.round(num(x.confidence)*100)}% confidence</span></div><div class="notice"><b>You said:</b> ${esc(transcript||'')}</div><div class="voice-review mt">${details.length?details.join(''):'<div class="notice error">Gemini did not find a loggable item. Cancel and try again with more detail.</div>'}</div>${x.summary?`<p class="small muted">${esc(x.summary)}</p>`:''}<div class="row mt"><button class="btn teal" onclick="confirmVoiceLog()" ${details.length?'':'disabled'}>Confirm & log</button><button class="btn light" onclick="closeModal()">Cancel</button></div>`);
  };
  window.confirmVoiceLog=function(){let x=window.pendingVoiceLog;if(!x)return;let d=x.date||today();if(!/^\d{4}-\d{2}-\d{2}$/.test(d))d=today();
    (x.meals||[]).forEach(m=>state.meals.push({id:uid('M'),date:d,name:m.food_name||'Voice meal',type:m.meal_type||'Other',calories:num(m.calories),protein:num(m.protein_g),carbs:num(m.carbs_g),fat:num(m.fat_g),fiber:num(m.fiber_g),source:'Gemini Voice',confidence:num(m.confidence)||num(x.confidence),note:m.notes||''}));
    let ws=x.workout_session||{},exs=ws.exercises||[];if(exs.length){let session=uid('S'),loggedAt=new Date().toISOString();exs.forEach((e,i)=>{let lib=allExercises().find(q=>String(q.name).toLowerCase()===String(e.exercise||'').toLowerCase())||{};let sets=num(e.sets),reps=num(e.reps),load=num(e.weight_lb),metricName=e.metric_name||lib.metric||'',metricValue=num(e.metric_value);if(!metricValue){if(String(metricName).toLowerCase()==='time')metricValue=num(e.time_seconds);else if(String(metricName).toLowerCase()==='distance')metricValue=num(e.distance_yards)||num(e.distance_miles);else if(String(metricName).toLowerCase()==='load')metricValue=load;}state.workouts.push({id:uid('W'),date:d,sessionId:session,plan:'Voice Log',workoutType:ws.workout_type||'Voice',source:'Gemini Voice',loggedAt,exercise:e.exercise||'Voice exercise',category:lib.category||e.category||'',sets,reps,load,volume:sets*reps*load,duration:i===0?num(ws.duration_min):0,cardio:i===0?num(ws.cardio_min):0,caloriesBurned:i===0?num(ws.calories_burned_kcal):0,distance:num(e.distance_yards)||num(e.distance_miles),time:num(e.time_seconds),metricName,metricValue,unit:e.metric_unit||lib.unit||'',rpe:num(e.rpe),recovery:num(e.recovery_seconds),notes:e.notes||ws.notes||'',completed:true})})}
    if(num(x.water_oz))state.water.push({id:uid('H'),date:d,oz:num(x.water_oz),time:new Date().toTimeString().slice(0,5),source:'Voice'});
    let b=x.body||{};if(num(b.weight_lb)||num(b.sleep_hours)||num(b.energy_score)||num(b.resting_hr_bpm)){let existing=state.body.find(q=>q.date===d),entry=existing||{id:uid('B'),date:d,weight:0,sleep:0,energy:0,hr:0};if(num(b.weight_lb))entry.weight=num(b.weight_lb);if(num(b.sleep_hours))entry.sleep=num(b.sleep_hours);if(num(b.energy_score))entry.energy=num(b.energy_score);if(num(b.resting_hr_bpm))entry.hr=num(b.resting_hr_bpm);if(!existing)state.body.push(entry)}
    save();window.pendingVoiceLog=null;window.dailyActivityDate=d;closeModal();toast('Voice log saved');render();
  };

  window.healthProgressHtml=function(){let vals=Object.keys(state.healthHistory||{}).sort().slice(-30).map(k=>state.healthHistory[k]).filter(Boolean);if(!vals.length)return `<div class="card mt"><b>Health Connect trends</b><p class="small muted">No synced Health Connect history yet. Sync from Today or Settings to start including watch/health stats in reports.</p></div>`;let avg=k=>{let a=vals.map(x=>num(x[k])).filter(v=>v>0);return a.length?a.reduce((x,y)=>x+y,0)/a.length:0};let latest=vals[vals.length-1]||{};return `<div class="card mt"><b>Health Connect trends</b><div class="grid3 mt"><div><div class="tiny muted">Avg daily HR</div><b>${fmt(avg('average_heart_rate_bpm'))} bpm</b></div><div><div class="tiny muted">Avg resting HR</div><b>${fmt(avg('resting_hr_bpm'))} bpm</b></div><div><div class="tiny muted">Avg steps</div><b>${fmt(avg('steps'))}</b></div><div><div class="tiny muted">Latest HR</div><b>${num(latest.latest_heart_rate_bpm)?fmt(latest.latest_heart_rate_bpm)+' bpm':'—'}</b></div><div><div class="tiny muted">Avg active kcal</div><b>${fmt(avg('active_calories_kcal'))}</b></div><div><div class="tiny muted">Avg total kcal</div><b>${fmt(avg('total_calories_kcal'))}</b></div><div><div class="tiny muted">Avg sleep</div><b>${fmt(avg('sleep_hours'),1)} hr</b></div><div><div class="tiny muted">Avg distance</div><b>${fmt(avg('distance_miles'),2)} mi</b></div></div></div>`};

  window.healthSummary=function(days=7){let from=dateDaysAgo(days-1),rows=Object.keys(state.healthHistory||{}).filter(k=>k>=from).sort().map(k=>state.healthHistory[k]);let avg=k=>{let a=rows.map(x=>num(x[k])).filter(v=>v>0);return a.length?a.reduce((x,y)=>x+y,0)/a.length:0};return {days,daysSynced:rows.length,averageHeartRate:avg('average_heart_rate_bpm'),averageSteps:avg('steps'),averageRestingHr:avg('resting_hr_bpm'),averageSleepHours:avg('sleep_hours'),averageActiveCalories:avg('active_calories_kcal'),averageTotalCalories:avg('total_calories_kcal'),averageDistanceMiles:avg('distance_miles'),latestHeartRate:rows.length?num(rows[rows.length-1].latest_heart_rate_bpm):0}};
  window.weeklyReport=function(){let base=oldWeeklyReport();let h=healthSummary(7);return base+(h.daysSynced?` Health Connect (${h.daysSynced} synced day${h.daysSynced===1?'':'s'}): <b>${fmt(h.averageHeartRate)}</b> bpm avg daily HR, <b>${fmt(h.averageRestingHr)}</b> bpm avg resting HR, <b>${fmt(h.averageSteps)}</b> avg steps, and <b>${fmt(h.averageActiveCalories)}</b> avg active kcal burned.`:'')};
  window.askAiInsight=function(){if(!native())return toast('Native Gemini bridge unavailable');$('#aiInsight').innerHTML='<div class="notice">Generating a progress insight…</div>';let payload={sevenDay:summary(7),thirtyDay:summary(30),weekly:weekly(),bests:(window.BENCHMARKS||[]).map(bestBenchmark),goals:state.goals,healthConnect7Day:healthSummary(7),healthConnect30Day:healthSummary(30)};Android.generateInsight(JSON.stringify(payload))};

  window.onNativeResult=function(raw){
    let r;try{r=typeof raw==='string'?JSON.parse(raw):raw}catch(e){return oldNativeResult(raw)}
    if(r.kind==='speech_partial'){let el=$('#voiceStatus');if(el)el.textContent='Listening: '+(r.text||'');return}
    if(r.kind==='speech_transcript'){let el=$('#voiceStatus');if(el)el.textContent='Heard: '+(r.text||'');parseVoiceTranscript(r.text||'',r.mode||'auto');return}
    if(r.kind==='speech_error'){let el=$('#voiceStatus');if(el)el.textContent=r.message||'Voice recognition failed';toast(r.message||'Voice recognition failed');return}
    if(r.kind==='voice_parsed'){window.pendingVoiceLog=r.data||{};showVoiceReview(window.pendingVoiceLog,r.transcript||'');return}
    if(r.kind==='health_permissions'){let el=$('#healthStatus');if(el)el.textContent=r.granted?'Health Connect access granted. Syncing…':'Health Connect access was not fully granted.';if(r.granted)syncHealthDate(window.pendingHealthDate||today());return}
    if(r.kind==='health_data'){let x=r.data||{};x.synced_at=new Date().toISOString();let date=x.date||today();state.healthHistory[date]=x;save();window.pendingHealthDate=null;let el=$('#healthStatus');if(el)el.textContent='Health Connect synced successfully.';toast('Health stats synced for '+dateLabel(date));render();return}
    if(r.kind==='health_error'){let el=$('#healthStatus');if(el)el.textContent=r.message||'Health Connect error';toast(r.message||'Health Connect error');return}
    if(r.kind==='backup_import'){
      try{let x=JSON.parse(r.data);if(x&&x.workouts&&x.meals){state=x;state.healthHistory=state.healthHistory||{};state.healthSyncEnabled=state.healthSyncEnabled!==false;state.version=Math.max(Number(state.version)||3,5);save();toast('Backup imported');render()}else toast('Backup format not recognized')}catch(e){toast('Backup import failed')}return;
    }
    return oldNativeResult(raw);
  };

  render();
})();

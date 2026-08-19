/* Fitness Command Center v1.1.5 add-on: retroactive meal/workout logging + editing */
(function(){
  state.version=Math.max(Number(state.version)||3,7);
  save();

  const baseRenderTraining=window.renderTraining;
  const baseRenderPlan=window.renderPlan;
  const baseStartVoiceLog=window.startVoiceLog;
  const baseConfirmVoiceLog=window.confirmVoiceLog;
  const baseAnalyzeMeal=window.analyzeMeal;
  const baseDailyActivityHtml=window.dailyActivityHtml;

  window.entryDate=window.entryDate||today();
  window.voiceTargetDate=null;
  window.pendingMealTargetDate=null;

  window.validEntryDate=function(d){
    if(!/^\d{4}-\d{2}-\d{2}$/.test(String(d||'')))return today();
    return d>today()?today():d;
  };
  window.entryDateLabel=function(d=window.entryDate){return d===today()?'Today':dateLabel(d)};
  window.setEntryDate=function(d){
    window.entryDate=validEntryDate(d);
    if(page==='food'||page==='training')render();
  };
  window.shiftEntryDate=function(delta){
    let d=new Date(validEntryDate(window.entryDate)+'T12:00:00');
    d.setDate(d.getDate()+delta);
    window.entryDate=validEntryDate(localISODate(d));
    if(page==='food'||page==='training')render();
  };
  window.historyDateBar=function(title){
    let d=validEntryDate(window.entryDate),past=d!==today();
    return `<div class="card mb" style="background:var(--bluebg)"><div class="between"><div><div class="eyebrow">${esc(title||'Log date')}</div><b>${esc(entryDateLabel(d))}</b></div><span class="pill ${past?'warn':'green'}">${past?'Past-day editing':'Current day'}</span></div><div class="row wrap mt"><button class="btn sm light" onclick="shiftEntryDate(-1)">← Day</button><input type="date" value="${d}" max="${today()}" style="width:auto;padding:7px 8px" onchange="setEntryDate(this.value)"><button class="btn sm light" onclick="shiftEntryDate(1)" ${d===today()?'disabled':''}>Day →</button><button class="btn sm teal" onclick="setEntryDate('${today()}')">Today</button></div><div class="tiny muted mt">Anything you add, edit, photograph, or dictate on this screen is saved to <b>${esc(entryDateLabel(d))}</b>.</div></div>`;
  };

  window.openFoodForDate=function(d){window.entryDate=validEntryDate(d);setPage('food')};
  window.openTrainingForDate=function(d){window.entryDate=validEntryDate(d);setPage('training')};
  window.startVoiceForDate=function(d,mode='auto'){
    window.voiceTargetDate=validEntryDate(d);
    if(!baseStartVoiceLog)return toast('Voice bridge unavailable');
    baseStartVoiceLog(mode);
  };

  window.startVoiceLog=function(mode='auto'){
    window.voiceTargetDate=(page==='food'||page==='training')?validEntryDate(window.entryDate):today();
    return baseStartVoiceLog?baseStartVoiceLog(mode):toast('Voice bridge unavailable');
  };
  window.confirmVoiceLog=function(){
    let target=window.voiceTargetDate?validEntryDate(window.voiceTargetDate):null;
    if(target&&window.pendingVoiceLog){window.pendingVoiceLog.date=target;window.entryDate=target;}
    let out=baseConfirmVoiceLog?baseConfirmVoiceLog():null;
    window.voiceTargetDate=null;
    return out;
  };

  window.analyzeMeal=function(){
    window.pendingMealTargetDate=validEntryDate(window.entryDate);
    return baseAnalyzeMeal?baseAnalyzeMeal():toast('Meal analysis unavailable');
  };

  window.renderFood=function(){
    let date=validEntryDate(window.entryDate),d=dayData(date);
    $('#content').innerHTML=`${historyDateBar('Nutrition log date')}<div class="eyebrow">Nutrition</div><div class="section-title">Food & meal analysis</div>
      <div class="card mb"><div class="between"><b>Speak a meal</b><span class="pill">🎙</span></div><p class="small muted">Tell the app what you ate and any portions you know. Gemini estimates missing nutrition values, then shows them for approval. The selected date above is used.</p><button class="btn block teal" onclick="startVoiceLog('food')">🎙 Log food by voice for ${esc(entryDateLabel(date))}</button></div>
      <div class="grid">${cardKpi('Calories',fmt(d.calories),`${fmt(state.goals.calories)} goal`)}${cardKpi('Protein',fmt(d.protein)+'g',`${fmt(state.goals.protein)}g goal`)}${cardKpi('Carbs',fmt(d.carbs)+'g')}${cardKpi('Fat',fmt(d.fat)+'g')}</div>
      <div class="card mt"><div class="between"><b>Analyze a meal photo</b><span class="pill">Gemini 3.5 Flash-Lite</span></div><p class="small muted">Choose a photo already on your phone. The image is sent only when you tap Analyze. Approved results are saved to ${esc(entryDateLabel(date))}.</p><input id="mealPhoto" type="file" accept="image/*"><img id="mealPreview" class="photo mt"><label class="mt"><span>Meal type</span><select id="mealType"><option>Breakfast</option><option>Lunch</option><option>Dinner</option><option>Snack</option><option>Pre-Workout</option><option>Post-Workout</option><option>Drink</option><option>Other</option></select></label><button class="btn block teal" onclick="analyzeMeal()">Analyze selected photo</button><div id="mealAiStatus" class="small muted mt"></div><div id="mealResult"></div></div>
      <div class="card mt"><b>Manual meal</b><div class="form2 mt"><label><span>Food / meal</span><input id="mfName"></label><label><span>Meal type</span><select id="mfType"><option>Breakfast</option><option>Lunch</option><option>Dinner</option><option>Snack</option><option>Other</option></select></label><label><span>Calories</span><input id="mfCal" inputmode="decimal"></label><label><span>Protein g</span><input id="mfPro" inputmode="decimal"></label><label><span>Carbs g</span><input id="mfCarb" inputmode="decimal"></label><label><span>Fat g</span><input id="mfFat" inputmode="decimal"></label><label><span>Fiber g</span><input id="mfFiber" inputmode="decimal"></label></div><button class="btn block light" onclick="saveManualMeal()">Save meal to ${esc(entryDateLabel(date))}</button></div>
      <div class="section-title">Meals — ${esc(entryDateLabel(date))}</div><div class="card">${mealListHtml(date)}</div>`;
    let inp=$('#mealPhoto');inp&&inp.addEventListener('change',()=>{let f=inp.files&&inp.files[0];if(f){let img=$('#mealPreview');img.src=URL.createObjectURL(f);img.style.display='block';$('#mealAiStatus').textContent='Photo selected. Tap Analyze when ready.'}});
  };

  window.mealListHtml=function(date){
    let rows=(state.meals||[]).filter(m=>m.date===date);
    if(!rows.length)return '<div class="muted small">No meals logged for this day.</div>';
    return rows.map(m=>`<div class="statline"><div><b>${esc(m.name)}</b><div class="tiny muted">${esc(m.type||'')} • ${esc(m.source||'Manual')}</div><div class="tiny muted">P ${fmt(m.protein)} • C ${fmt(m.carbs)} • F ${fmt(m.fat)} • Fiber ${fmt(m.fiber)}g</div></div><div style="text-align:right"><b>${fmt(m.calories)} kcal</b><div class="row wrap" style="justify-content:flex-end;margin-top:5px"><button class="btn sm light" onclick="editMeal('${m.id}')">Edit</button><button class="btn sm danger" onclick="deleteMeal('${m.id}')">Delete</button></div></div></div>`).join('');
  };
  window.saveManualMeal=function(){
    let name=$('#mfName').value.trim();if(!name)return toast('Enter a food/meal name');
    let date=validEntryDate(window.entryDate);
    state.meals.push({id:uid('M'),date,name,type:$('#mfType').value,calories:num($('#mfCal').value),protein:num($('#mfPro').value),carbs:num($('#mfCarb').value),fat:num($('#mfFat').value),fiber:num($('#mfFiber').value),source:'Manual',loggedAt:new Date().toISOString()});
    save();renderFood();toast(`Meal saved to ${entryDateLabel(date)}`);
  };
  window.saveAiMeal=function(){
    if(!pendingMeal)return;
    let date=validEntryDate(window.pendingMealTargetDate||window.entryDate);
    state.meals.push({id:uid('AI'),date,name:(pendingMeal.estimated_foods||[]).join('; '),type:$('#mealType')?.value||'Meal',calories:num(pendingMeal.estimated_calories),protein:num(pendingMeal.estimated_protein_g),carbs:num(pendingMeal.estimated_carbs_g),fat:num(pendingMeal.estimated_fat_g),fiber:num(pendingMeal.estimated_fiber_g),source:'Gemini photo estimate',confidence:num(pendingMeal.confidence),note:pendingMeal.short_explanation,loggedAt:new Date().toISOString()});
    save();pendingMeal=null;window.pendingMealTargetDate=null;window.entryDate=date;renderFood();toast(`Meal saved to ${entryDateLabel(date)}`);
  };
  window.editMeal=function(id){
    let m=(state.meals||[]).find(x=>x.id===id);if(!m)return;
    showModal(`<div class="eyebrow">Edit meal</div><div class="section-title">${esc(m.name||'Meal')}</div><label><span>Date</span><input id="emDate" type="date" max="${today()}" value="${esc(m.date||today())}"></label><div class="form2"><label><span>Food / meal</span><input id="emName" value="${esc(m.name||'')}"></label><label><span>Meal type</span><input id="emType" value="${esc(m.type||'Other')}"></label><label><span>Calories</span><input id="emCal" inputmode="decimal" value="${esc(m.calories||'')}"></label><label><span>Protein g</span><input id="emPro" inputmode="decimal" value="${esc(m.protein||'')}"></label><label><span>Carbs g</span><input id="emCarb" inputmode="decimal" value="${esc(m.carbs||'')}"></label><label><span>Fat g</span><input id="emFat" inputmode="decimal" value="${esc(m.fat||'')}"></label><label><span>Fiber g</span><input id="emFiber" inputmode="decimal" value="${esc(m.fiber||'')}"></label></div><button class="btn block teal" onclick="saveMealEdit('${m.id}')">Save changes</button><button class="btn block light mt" onclick="closeModal()">Cancel</button>`);
  };
  window.saveMealEdit=function(id){
    let m=(state.meals||[]).find(x=>x.id===id);if(!m)return;
    let date=validEntryDate($('#emDate').value),name=$('#emName').value.trim();if(!name)return toast('Enter a food/meal name');
    Object.assign(m,{date,name,type:$('#emType').value.trim()||'Other',calories:num($('#emCal').value),protein:num($('#emPro').value),carbs:num($('#emCarb').value),fat:num($('#emFat').value),fiber:num($('#emFiber').value),editedAt:new Date().toISOString()});
    window.entryDate=date;save();closeModal();renderFood();toast('Meal updated');
  };
  window.deleteMeal=function(id){
    let m=(state.meals||[]).find(x=>x.id===id);if(!m)return;
    if(!confirm(`Delete ${m.name||'this meal'} from ${dateLabel(m.date)}?`))return;
    state.meals=state.meals.filter(x=>x.id!==id);save();render();toast('Meal deleted');
  };

  window.showManualWorkout=function(date=window.entryDate){
    date=validEntryDate(date);let names=allExercises().map(x=>x.name).sort();
    showModal(`<div class="eyebrow">Workout log</div><div class="section-title">Log an exercise / session</div><label><span>Date</span><input id="mwDate" type="date" max="${today()}" value="${date}"></label><label><span>Exercise</span><select id="mwEx">${names.map(n=>`<option>${esc(n)}</option>`).join('')}</select></label><div class="form2"><label><span>Sets</span><input id="mwSets" inputmode="decimal"></label><label><span>Reps</span><input id="mwReps" inputmode="decimal"></label><label><span>Load (lb)</span><input id="mwLoad" inputmode="decimal"></label><label><span>Duration (min)</span><input id="mwDur" inputmode="decimal"></label><label><span>Cardio (min)</span><input id="mwCardio" inputmode="decimal"></label><label><span>Distance (yd)</span><input id="mwDist" inputmode="decimal"></label><label><span>Time (sec)</span><input id="mwTime" inputmode="decimal"></label><label><span>RPE 1-10</span><input id="mwRpe" inputmode="decimal"></label><label><span>Recovery 1-5</span><input id="mwRec" inputmode="decimal"></label><label><span>Calories burned (optional)</span><input id="mwBurn" inputmode="decimal" placeholder="Watch/device or known value"></label></div><label><span>Notes</span><textarea id="mwNotes"></textarea></label><button class="btn block teal" onclick="saveManualWorkout()">Save workout entry</button><button class="btn block light mt" onclick="closeModal()">Cancel</button>`);
  };
  window.saveManualWorkout=function(){
    let date=validEntryDate($('#mwDate').value||window.entryDate),name=$('#mwEx').value,e=allExercises().find(x=>x.name===name)||{},sets=num($('#mwSets').value),reps=num($('#mwReps').value),load=num($('#mwLoad').value),time=num($('#mwTime').value),dist=num($('#mwDist').value);let metricValue=0,metricName=e.metric||'';
    if((e.metric||'').toLowerCase()==='time')metricValue=time;else if((e.metric||'').toLowerCase()==='distance')metricValue=dist;else if((e.metric||'').toLowerCase()==='load')metricValue=load;
    state.workouts.push({id:uid('W'),date,sessionId:uid('S'),plan:'Manual',workoutType:'Manual',source:'Manual',loggedAt:new Date().toISOString(),exercise:name,category:e.category||'',sets,reps,load,volume:sets*reps*load,duration:num($('#mwDur').value),cardio:num($('#mwCardio').value),distance:dist,time,metricName,metricValue,unit:e.unit||'',rpe:num($('#mwRpe').value),recovery:num($('#mwRec').value),caloriesBurned:num($('#mwBurn').value),notes:$('#mwNotes').value,completed:true});
    window.entryDate=date;save();closeModal();toast(`Workout saved to ${entryDateLabel(date)}`);renderTraining();
  };

  window.quickTest=function(name,metric,unit){
    let date=validEntryDate(window.entryDate);
    showModal(`<div class="section-title">Log ${esc(name)}</div><label><span>Date</span><input id="qtDate" type="date" max="${today()}" value="${date}"></label><label><span>${esc(metric)} (${esc(unit)})</span><input id="qtVal" inputmode="decimal"></label><div class="form2"><label><span>RPE 1-10</span><input id="qtRpe" inputmode="decimal"></label><label><span>Recovery 1-5</span><input id="qtRec" inputmode="decimal"></label></div><button class="btn block teal" onclick="saveQuickTest('${esc(name).replace(/'/g,"\\'")}','${esc(metric)}','${esc(unit)}')">Save result</button>`);
  };
  window.saveQuickTest=function(name,metric,unit){
    let date=validEntryDate($('#qtDate').value||window.entryDate),ex=allExercises().find(x=>x.name===name)||{};
    state.workouts.push({id:uid('T'),date,sessionId:uid('TEST'),workoutType:'Performance Test',source:'Performance Test',loggedAt:new Date().toISOString(),exercise:name,category:ex.category||'',metricName:metric,metricValue:num($('#qtVal').value),unit,rpe:num($('#qtRpe').value),recovery:num($('#qtRec').value),duration:0,completed:true});
    window.entryDate=date;save();closeModal();renderTraining();toast(`Result saved to ${entryDateLabel(date)}`);
  };

  window.editWorkout=function(id){
    let x=(state.workouts||[]).find(q=>q.id===id);if(!x)return;
    let names=allExercises().map(e=>e.name).sort();if(x.exercise&&!names.includes(x.exercise))names.unshift(x.exercise);
    showModal(`<div class="eyebrow">Edit exercise</div><div class="section-title">${esc(x.exercise||'Workout entry')}</div><label><span>Date</span><input id="ewDate" type="date" max="${today()}" value="${esc(x.date||today())}"></label><label><span>Exercise</span><select id="ewEx">${names.map(n=>`<option ${n===x.exercise?'selected':''}>${esc(n)}</option>`).join('')}</select></label><div class="form2"><label><span>Sets</span><input id="ewSets" inputmode="decimal" value="${esc(x.sets||'')}"></label><label><span>Reps</span><input id="ewReps" inputmode="decimal" value="${esc(x.reps||'')}"></label><label><span>Load (lb)</span><input id="ewLoad" inputmode="decimal" value="${esc(x.load||'')}"></label><label><span>Duration (min)</span><input id="ewDur" inputmode="decimal" value="${esc(x.duration||'')}"></label><label><span>Cardio (min)</span><input id="ewCardio" inputmode="decimal" value="${esc(x.cardio||'')}"></label><label><span>Distance</span><input id="ewDist" inputmode="decimal" value="${esc(x.distance||'')}"></label><label><span>Time (sec)</span><input id="ewTime" inputmode="decimal" value="${esc(x.time||'')}"></label><label><span>Result value</span><input id="ewMetric" inputmode="decimal" value="${esc(x.metricValue||'')}"></label><label><span>Result unit</span><input id="ewUnit" value="${esc(x.unit||'')}"></label><label><span>RPE 1-10</span><input id="ewRpe" inputmode="decimal" value="${esc(x.rpe||'')}"></label><label><span>Recovery</span><input id="ewRec" inputmode="decimal" value="${esc(x.recovery||'')}"></label><label><span>Calories burned</span><input id="ewBurn" inputmode="decimal" value="${esc(x.caloriesBurned||'')}"></label></div><label><span>Notes</span><textarea id="ewNotes">${esc(x.notes||'')}</textarea></label><button class="btn block teal" onclick="saveWorkoutEdit('${x.id}')">Save changes</button><button class="btn block light mt" onclick="closeModal()">Cancel</button>`);
  };
  window.saveWorkoutEdit=function(id){
    let x=(state.workouts||[]).find(q=>q.id===id);if(!x)return;
    let date=validEntryDate($('#ewDate').value),name=$('#ewEx').value,e=allExercises().find(q=>q.name===name)||{},sets=num($('#ewSets').value),reps=num($('#ewReps').value),load=num($('#ewLoad').value),time=num($('#ewTime').value),dist=num($('#ewDist').value),metricName=x.metricName||e.metric||'',metricValue=num($('#ewMetric').value);
    if(!metricValue){if(String(metricName).toLowerCase()==='time')metricValue=time;else if(String(metricName).toLowerCase()==='distance')metricValue=dist;else if(String(metricName).toLowerCase()==='load')metricValue=load;}
    Object.assign(x,{date,exercise:name,category:e.category||x.category||'',sets,reps,load,volume:sets*reps*load,duration:num($('#ewDur').value),cardio:num($('#ewCardio').value),distance:dist,time,metricName,metricValue,unit:$('#ewUnit').value.trim()||e.unit||x.unit||'',rpe:num($('#ewRpe').value),recovery:num($('#ewRec').value),caloriesBurned:num($('#ewBurn').value),notes:$('#ewNotes').value,editedAt:new Date().toISOString(),completed:true});
    window.entryDate=date;window.dailyActivityDate=date;save();closeModal();render();toast('Exercise updated');
  };
  window.deleteWorkout=function(id){
    let x=(state.workouts||[]).find(q=>q.id===id);if(!x)return;
    if(!confirm(`Delete ${x.exercise||'this exercise'} from ${dateLabel(x.date)}?`))return;
    state.workouts=state.workouts.filter(q=>q.id!==id);save();render();toast('Exercise deleted');
  };

  window.localWorkoutHtml=function(date){
    let groups=sessionGroupsFor(date);if(!groups.length)return '<div class="small muted">No exercises were logged for this day.</div>';
    return groups.map(rows=>{let first=rows[0]||{},burn=sessionBurn(rows),dur=sessionDuration(rows),src=workoutSourceLabel(first);return `<div class="exercise"><div class="between"><div><div class="name">${esc(first.workoutType||first.plan||'Workout session')}</div><div class="tiny muted">${esc(src)} • ${rows.length} exercise${rows.length===1?'':'s'}${dur?' • '+fmt(dur)+' min':''}</div></div>${burn?`<span class="pill green">${fmt(burn)} kcal burned</span>`:'<span class="pill">Logged</span>'}</div>${rows.map(x=>{let prs=prInfo(x);return `<div class="statline"><div><b>${esc(x.exercise||'Exercise')}</b><div class="tiny muted">${exerciseDetailText(x)}</div>${x.notes?`<div class="tiny muted">${esc(x.notes)}</div>`:''}</div><div class="row wrap">${prs.length?'<span class="pill green">PR</span>':''}<button class="btn sm light" onclick="editWorkout('${x.id}')">Edit</button><button class="btn sm danger" onclick="deleteWorkout('${x.id}')">Delete</button></div></div>`}).join('')}</div>`}).join('');
  };

  window.planRow=function(it,i){
    let date=validEntryDate(window.entryDate),ex=allExercises().find(x=>x.name===it[0])||{metric:'Quality',unit:'score'},done=state.workouts.some(x=>x.date===date&&x.plan===activePlan&&x.exercise===it[0]);
    return `<div class="checkrow"><input type="checkbox" id="pc${i}" ${done?'checked disabled':''}><div><div class="between"><b>${esc(it[0])}</b><div class="row">${done?'<span class="pill green">✓ logged</span>':''}<button class="btn sm light" onclick="event.preventDefault();showExerciseByName('${esc(it[0]).replace(/'/g,"\\'")}')">Info</button></div></div><div class="tiny muted">${esc(it[1])} • Rest ${esc(it[2])}</div><div class="metric-grid"><input id="psets${i}" inputmode="decimal" placeholder="Sets" ${done?'disabled':''}><input id="preps${i}" inputmode="decimal" placeholder="Reps" ${done?'disabled':''}><input id="pload${i}" inputmode="decimal" placeholder="Load lb" ${done?'disabled':''}></div><div class="metric-grid"><input id="pm${i}" inputmode="decimal" placeholder="${esc(ex.metric||'Result')} ${esc(ex.unit||'')}" ${done?'disabled':''}><input id="prpe${i}" inputmode="decimal" placeholder="RPE 1-10" ${done?'disabled':''}><input id="prec${i}" inputmode="decimal" placeholder="Recovery 1-5" ${done?'disabled':''}></div></div></div>`;
  };
  window.savePlanSession=function(){
    let p=TRAINING_PLANS[activePlan],sid=uid('S'),date=validEntryDate(window.entryDate),duration=num($('#planDuration').value),count=0;
    p.items.forEach((it,i)=>{let already=state.workouts.some(x=>x.date===date&&x.plan===activePlan&&x.exercise===it[0]);if($('#pc'+i).checked&&!already){let ex=allExercises().find(x=>x.name===it[0])||{},sets=num($('#psets'+i).value),reps=num($('#preps'+i).value),load=num($('#pload'+i).value);state.workouts.push({id:uid('W'),date,sessionId:sid,plan:activePlan,workoutType:activePlan,source:'Plan',loggedAt:new Date().toISOString(),exercise:it[0],category:ex.category||'',metricName:ex.metric||'',metricValue:num($('#pm'+i).value),unit:ex.unit||'',sets,reps,load,volume:sets*reps*load,rpe:num($('#prpe'+i).value),recovery:num($('#prec'+i).value),duration:count===0?duration:0,completed:true,notes:count===0?$('#planNotes').value:'',cardio:(ex.category==='Speed'||ex.category==='Agility')?duration:0});count++;}});
    save();activePlan=null;toast(count?`${count} exercises saved to ${entryDateLabel(date)}`:'No new exercises checked');renderTraining();
  };
  window.renderPlan=function(name){
    baseRenderPlan(name);let c=$('#content');if(c)c.insertAdjacentHTML('afterbegin',historyDateBar('Plan workout date'));
  };
  window.renderTraining=function(){
    baseRenderTraining();if(activePlan)return;let c=$('#content');if(!c)return;let date=validEntryDate(window.entryDate);
    c.insertAdjacentHTML('afterbegin',historyDateBar('Training log date'));
    let existing=(state.workouts||[]).filter(x=>x.date===date&&x.completed!==false);
    c.insertAdjacentHTML('beforeend',`<div class="card mt"><div class="between"><b>Exercises already logged — ${esc(entryDateLabel(date))}</b><span class="pill">${fmt(existing.length)} entries</span></div><div class="mt">${localWorkoutHtml(date)}</div></div>`);
  };

  window.dailyActivityHtml=function(date=window.dailyActivityDate||today()){
    let base=baseDailyActivityHtml(date),d=dayData(date);
    return base+`<div class="card mb"><div class="between"><div><b>Add or correct this day</b><div class="tiny muted">Past days are editable. Add forgotten items or correct/delete existing entries.</div></div><span class="pill">${esc(dateLabel(date))}</span></div><div class="grid3 mt"><button class="btn gold" onclick="openFoodForDate('${date}')">Meals (${fmt(d.meals.length)})</button><button class="btn teal" onclick="openTrainingForDate('${date}')">Exercises (${fmt(d.workouts.length)})</button><button class="btn light" onclick="startVoiceForDate('${date}','auto')">🎙 Voice log</button></div></div>`;
  };

  const priorRefreshDailyActivity=window.refreshDailyActivity;
  window.refreshDailyActivity=function(){if(priorRefreshDailyActivity)priorRefreshDailyActivity()};
})();

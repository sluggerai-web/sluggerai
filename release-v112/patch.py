from pathlib import Path
import sys

root=Path(sys.argv[1] if len(sys.argv)>1 else 'fitness-android')

# Version bump.
p=root/'app/build.gradle'
s=p.read_text()
s=s.replace('versionCode 4','versionCode 6').replace("versionName '1.1.0'","versionName '1.1.2'")
p.write_text(s)

# Android manifest launcher icon references.
p=root/'app/src/main/AndroidManifest.xml'
s=p.read_text()
if 'android:icon=' not in s:
    s=s.replace(
        'android:label="Fitness Command Center"\n        android:theme="@style/AppTheme">',
        'android:label="Fitness Command Center"\n        android:icon="@mipmap/ic_launcher"\n        android:roundIcon="@mipmap/ic_launcher_round"\n        android:theme="@style/AppTheme">'
    )
p.write_text(s)

# MainActivity: historical date sync, avg HR, exercise-session HR/calories, voice calories.
p=root/'app/src/main/java/com/david/fitnesscommandcenter/v2/MainActivity.kt'
s=p.read_text()
s=s.replace('@JavascriptInterface fun readHealthToday() { scope.launch { readHealthToday_() } }\n        @JavascriptInterface fun openHealthConnectSettings()',
'''@JavascriptInterface fun readHealthToday() { scope.launch { readHealthDate_(LocalDate.now()) } }
        @JavascriptInterface fun readHealthDate(date: String) {
            scope.launch {
                try { readHealthDate_(LocalDate.parse(date)) }
                catch (_: Exception) { sendResult(JSONObject().put("kind", "health_error").put("message", "Invalid health date.")) }
            }
        }
        @JavascriptInterface fun openHealthConnectSettings()''')
s=s.replace('For workouts, extract sets, reps, weight in pounds, duration, cardio minutes, distance, sprint/agility time, RPE, and recovery when spoken. Unknown numeric fields must be 0.',
'''For workouts, extract sets, reps, weight in pounds, duration, cardio minutes, distance, sprint/agility time, RPE, recovery, and calories burned when explicitly spoken. Unknown numeric fields must be 0. Do not invent calories burned; leave calories_burned_kcal as 0 unless the user stated it.''')
s=s.replace('"workout_type":{"type":"STRING"},"duration_min":{"type":"NUMBER"},"cardio_min":{"type":"NUMBER"},"notes":{"type":"STRING"},',
'''"workout_type":{"type":"STRING"},"duration_min":{"type":"NUMBER"},"cardio_min":{"type":"NUMBER"},"calories_burned_kcal":{"type":"NUMBER"},"notes":{"type":"STRING"},''')
s=s.replace('"required":["workout_type","duration_min","cardio_min","notes","exercises"]}',
'''"required":["workout_type","duration_min","cardio_min","calories_burned_kcal","notes","exercises"]}''')
start=s.find('    private suspend fun readHealthToday_() {')
end=s.find('    private fun healthStatus_()', start)
if start<0 or end<0:
    raise SystemExit('Could not locate readHealthToday_ block')
new_method=r'''    private suspend fun readHealthDate_(targetDate: LocalDate) {
        if (healthStatus_() != HealthConnectClient.SDK_AVAILABLE) {
            sendResult(JSONObject().put("kind", "health_error").put("message", "Health Connect is not available.")); return
        }
        try {
            val granted = healthClient.permissionController.getGrantedPermissions()
            if (!granted.containsAll(healthPermissions)) {
                withContext(Dispatchers.Main) { healthPermissionLauncher.launch(healthPermissions) }
                return
            }

            val zone = ZoneId.systemDefault()
            val start = targetDate.atStartOfDay(zone).toInstant()
            val now = Instant.now()
            val todayLocal = LocalDate.now(zone)
            val end = if (targetDate == todayLocal) now else targetDate.plusDays(1).atStartOfDay(zone).toInstant()
            if (start.isAfter(now)) {
                sendResult(JSONObject().put("kind", "health_error").put("message", "Health data cannot be read for a future date.")); return
            }
            val previousDay = start.minus(Duration.ofHours(18))
            val weightStart = start.minus(Duration.ofDays(90))

            val hrRecords = healthClient.readRecords(ReadRecordsRequest(HeartRateRecord::class, TimeRangeFilter.between(start, end))).records
            val latestSample = hrRecords.flatMap { it.samples }.maxByOrNull { it.time }
            val latestHr = latestSample?.beatsPerMinute ?: 0L

            val restingRecords = healthClient.readRecords(ReadRecordsRequest(RestingHeartRateRecord::class, TimeRangeFilter.between(start, end))).records
            val restingHr = restingRecords.maxByOrNull { it.time }?.beatsPerMinute ?: 0L

            val aggregation = healthClient.aggregate(
                AggregateRequest(
                    metrics = setOf(
                        StepsRecord.COUNT_TOTAL,
                        ActiveCaloriesBurnedRecord.ACTIVE_CALORIES_TOTAL,
                        TotalCaloriesBurnedRecord.ENERGY_TOTAL,
                        DistanceRecord.DISTANCE_TOTAL,
                        HeartRateRecord.BPM_AVG,
                        HeartRateRecord.BPM_MIN,
                        HeartRateRecord.BPM_MAX
                    ),
                    timeRangeFilter = TimeRangeFilter.between(start, end)
                )
            )
            val steps = aggregation[StepsRecord.COUNT_TOTAL] ?: 0L
            val activeCalories = aggregation[ActiveCaloriesBurnedRecord.ACTIVE_CALORIES_TOTAL]?.inKilocalories ?: 0.0
            val totalCalories = aggregation[TotalCaloriesBurnedRecord.ENERGY_TOTAL]?.inKilocalories ?: 0.0
            val distanceMiles = aggregation[DistanceRecord.DISTANCE_TOTAL]?.inMiles ?: 0.0
            val averageHr = aggregation[HeartRateRecord.BPM_AVG] ?: 0L
            val minHr = aggregation[HeartRateRecord.BPM_MIN] ?: 0L
            val maxHr = aggregation[HeartRateRecord.BPM_MAX] ?: 0L

            val sleepRecords = healthClient.readRecords(ReadRecordsRequest(SleepSessionRecord::class, TimeRangeFilter.between(previousDay, end))).records
            val sleepSeconds = sleepRecords.sumOf { r ->
                val ss = if (r.startTime.isBefore(previousDay)) previousDay else r.startTime
                val ee = if (r.endTime.isAfter(end)) end else r.endTime
                if (ee.isAfter(ss)) Duration.between(ss, ee).seconds else 0L
            }
            val sleepHours = sleepSeconds / 3600.0

            val weightRecords = healthClient.readRecords(ReadRecordsRequest(WeightRecord::class, TimeRangeFilter.between(weightStart, end))).records
            val weightLb = weightRecords.maxByOrNull { it.time }?.weight?.inPounds ?: 0.0

            val exerciseRecords = healthClient.readRecords(ReadRecordsRequest(ExerciseSessionRecord::class, TimeRangeFilter.between(start, end))).records
            val exerciseDetails = JSONArray()
            for (record in exerciseRecords.sortedBy { it.startTime }) {
                val sessionStart = if (record.startTime.isBefore(start)) start else record.startTime
                val sessionEnd = if (record.endTime.isAfter(end)) end else record.endTime
                var sessionAvgHr = 0L
                var sessionActiveCalories = 0.0
                if (sessionEnd.isAfter(sessionStart)) {
                    val sessionAgg = healthClient.aggregate(
                        AggregateRequest(
                            metrics = setOf(
                                ActiveCaloriesBurnedRecord.ACTIVE_CALORIES_TOTAL,
                                HeartRateRecord.BPM_AVG
                            ),
                            timeRangeFilter = TimeRangeFilter.between(sessionStart, sessionEnd)
                        )
                    )
                    sessionAvgHr = sessionAgg[HeartRateRecord.BPM_AVG] ?: 0L
                    sessionActiveCalories = sessionAgg[ActiveCaloriesBurnedRecord.ACTIVE_CALORIES_TOTAL]?.inKilocalories ?: 0.0
                }
                exerciseDetails.put(
                    JSONObject()
                        .put("start_time", record.startTime.toString())
                        .put("end_time", record.endTime.toString())
                        .put("duration_min", round2_(Duration.between(record.startTime, record.endTime).seconds / 60.0))
                        .put("title", record.title ?: "")
                        .put("notes", record.notes ?: "")
                        .put("exercise_type", record.exerciseType)
                        .put("average_heart_rate_bpm", sessionAvgHr)
                        .put("active_calories_kcal", round2_(sessionActiveCalories))
                        .put("source_package", record.metadata.dataOrigin.packageName)
                )
            }

            val hrSources = JSONArray()
            hrRecords.map { it.metadata.dataOrigin.packageName }.filter { it.isNotBlank() }.distinct().forEach { hrSources.put(it) }

            val data = JSONObject()
                .put("date", targetDate.toString())
                .put("latest_heart_rate_bpm", latestHr)
                .put("average_heart_rate_bpm", averageHr)
                .put("min_heart_rate_bpm", minHr)
                .put("max_heart_rate_bpm", maxHr)
                .put("resting_hr_bpm", restingHr)
                .put("steps", steps)
                .put("active_calories_kcal", round2_(activeCalories))
                .put("total_calories_kcal", round2_(totalCalories))
                .put("distance_miles", round2_(distanceMiles))
                .put("sleep_hours", round2_(sleepHours))
                .put("weight_lb", round2_(weightLb))
                .put("exercise_sessions", exerciseRecords.size)
                .put("exercise_sessions_detail", exerciseDetails)
                .put("heart_rate_sources", hrSources)
                .put("source_label", "Health Connect")
            sendResult(JSONObject().put("kind", "health_data").put("data", data))
        } catch (e: Exception) {
            sendResult(JSONObject().put("kind", "health_error").put("message", "Could not read Health Connect: ${e.message}"))
        }
    }

'''
s=s[:start]+new_method+s[end:]
p.write_text(s)

# Web UI: manual calorie field + source/timestamp metadata for locally logged workouts.
p=root/'app/src/main/assets/index.html'
s=p.read_text()
s=s.replace('''<label><span>Recovery 1-5</span><input id="mwRec" inputmode="decimal"></label></div><label><span>Notes</span>''',
'''<label><span>Recovery 1-5</span><input id="mwRec" inputmode="decimal"></label><label><span>Calories burned (optional)</span><input id="mwBurn" inputmode="decimal" placeholder="Watch/device or known value"></label></div><label><span>Notes</span>''')
old="""state.workouts.push({id:uid('W'),date:today(),sessionId:uid('S'),plan:'Manual',workoutType:'Manual',exercise:name,category:e.category||'',sets,reps,load,volume:sets*reps*load,duration:num($('#mwDur').value),cardio:num($('#mwCardio').value),distance:dist,time,metricName,metricValue,unit:e.unit||'',rpe:num($('#mwRpe').value),recovery:num($('#mwRec').value),notes:$('#mwNotes').value,completed:true});"""
new="""state.workouts.push({id:uid('W'),date:today(),sessionId:uid('S'),plan:'Manual',workoutType:'Manual',source:'Manual',loggedAt:new Date().toISOString(),exercise:name,category:e.category||'',sets,reps,load,volume:sets*reps*load,duration:num($('#mwDur').value),cardio:num($('#mwCardio').value),distance:dist,time,metricName,metricValue,unit:e.unit||'',rpe:num($('#mwRpe').value),recovery:num($('#mwRec').value),caloriesBurned:num($('#mwBurn').value),notes:$('#mwNotes').value,completed:true});"""
if old not in s: raise SystemExit('Could not locate manual workout save block')
s=s.replace(old,new)
s=s.replace("state.workouts.push({id:uid('W'),date,sessionId:sid,plan:activePlan,workoutType:activePlan,exercise:it[0]",
            "state.workouts.push({id:uid('W'),date,sessionId:sid,plan:activePlan,workoutType:activePlan,source:'Plan',loggedAt:new Date().toISOString(),exercise:it[0]")
s=s.replace("state.workouts.push({id:uid('T'),date:today(),sessionId:uid('TEST'),workoutType:'Performance Test',exercise:name",
            "state.workouts.push({id:uid('T'),date:today(),sessionId:uid('TEST'),workoutType:'Performance Test',source:'Performance Test',loggedAt:new Date().toISOString(),exercise:name")
p.write_text(s)

# Launcher resources.
(root/'app/src/main/res/drawable').mkdir(parents=True,exist_ok=True)
(root/'app/src/main/res/mipmap-anydpi').mkdir(parents=True,exist_ok=True)
(root/'app/src/main/res/mipmap-anydpi-v26').mkdir(parents=True,exist_ok=True)
(root/'app/src/main/res/values').mkdir(parents=True,exist_ok=True)
(root/'app/src/main/res/values/colors.xml').write_text('''<resources>\n    <color name="launcher_background">#0B1F33</color>\n</resources>\n''')
icon='''<vector xmlns:android="http://schemas.android.com/apk/res/android"\n    android:width="108dp" android:height="108dp"\n    android:viewportWidth="108" android:viewportHeight="108">\n    <path android:fillColor="#1F7A8C" android:pathData="M54,18A36,36 0,1 0,54 90A36,36 0,1 0,54 18M54,27A27,27 0,1 1,54 81A27,27 0,1 1,54 27"/>\n    <path android:fillColor="#FFFFFF" android:pathData="M24,49H30V59H24ZM32,44H38V64H32ZM40,51H68V57H40ZM70,44H76V64H70ZM78,49H84V59H78Z"/>\n    <path android:fillColor="#D4AF37" android:pathData="M43,37L50,51L55,41L61,55L66,46L70,46L61,65L55,51L50,61L39,37Z"/>\n</vector>\n'''
(root/'app/src/main/res/drawable/ic_launcher_foreground.xml').write_text(icon)
(root/'app/src/main/res/mipmap-anydpi/ic_launcher.xml').write_text(icon)
(root/'app/src/main/res/mipmap-anydpi/ic_launcher_round.xml').write_text(icon)
adaptive='''<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">\n    <background android:drawable="@color/launcher_background" />\n    <foreground android:drawable="@drawable/ic_launcher_foreground" />\n</adaptive-icon>\n'''
(root/'app/src/main/res/mipmap-anydpi-v26/ic_launcher.xml').write_text(adaptive)
(root/'app/src/main/res/mipmap-anydpi-v26/ic_launcher_round.xml').write_text(adaptive)

print('Patched Fitness Command Center source to v1.1.2')

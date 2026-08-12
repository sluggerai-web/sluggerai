from pathlib import Path
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('fitness-android')

# Version bump over v1.1.2.
p = root / 'app/build.gradle'
s = p.read_text()
s = s.replace('versionCode 6', 'versionCode 7')
s = s.replace("versionName '1.1.2'", "versionName '1.1.3'")
p.write_text(s)

# Load the report add-on after the existing Health Connect/Voice add-on.
p = root / 'app/src/main/assets/index.html'
s = p.read_text()
if '<script src="reportaddon.js"></script>' not in s:
    s = s.replace('<script src="addon.js"></script></body></html>', '<script src="addon.js"></script><script src="reportaddon.js"></script></body></html>')
p.write_text(s)

# Add a dedicated Gemini report bridge while preserving the existing insight feature.
p = root / 'app/src/main/java/com/david/fitnesscommandcenter/v2/MainActivity.kt'
s = p.read_text()
bridge = '        @JavascriptInterface fun generateInsight(summary: String) { scope.launch { generateInsight_(summary) } }\n'
if '@JavascriptInterface fun generateReport' not in s:
    if bridge not in s:
        raise SystemExit('Could not locate generateInsight bridge')
    s = s.replace(bridge, bridge + '        @JavascriptInterface fun generateReport(request: String) { scope.launch { generateReport_(request) } }\n')

method = r'''
    private suspend fun generateReport_(request: String) {
        try {
            requireApiKey_()
            val req = JSONObject(request)
            val reportType = req.optString("report_type", "daily").lowercase(Locale.US)
            val label = req.optString("label", "Gemini Fitness Report")
            val startDate = req.optString("start_date", "")
            val endDate = req.optString("end_date", "")
            val requestId = req.optString("request_id", "")
            val auto = req.optBoolean("auto", false)
            val data = req.optJSONObject("data") ?: JSONObject()
            val maxWords = when (reportType) {
                "daily" -> 450
                "weekly" -> 700
                else -> 900
            }
            val nextLabel = when (reportType) {
                "daily" -> "tomorrow / the next training day"
                "weekly" -> "the next 7 days"
                else -> "the next month"
            }
            val prompt = """
              You are the coaching-report engine inside a private personal Fitness Command Center app.
              Generate a ${reportType.uppercase(Locale.US)} report titled: $label.
              Analyze ONLY the supplied tracking data. It may include meals/macros, workouts/exercises/sets/reps/load/performance results, water, habits, body/recovery, Health Connect / Pixel Watch average-resting-min-max heart rate, steps, sleep, active/total calories burned, distance, weight and prior-period comparison data.

              Use these sections in this order:
              OVERVIEW
              WHAT WENT WELL
              WHAT COULD BE BETTER
              TRAINING & PERFORMANCE
              NUTRITION
              HYDRATION
              HEART RATE & RECOVERY
              FOCUS FOR $nextLabel

              Requirements:
              - Use concrete numbers from the data whenever useful.
              - Compare with goals and the previous comparable period when the data supports it.
              - Mention meaningful consistency, personal bests, training volume and adherence when present.
              - For meals, discuss overall calories/macros/fiber and meal patterns without inventing foods or exact nutrient values.
              - Keep calories eaten separate from active calories burned and total calories burned. Do not claim a precise energy deficit/surplus from wearable estimates.
              - Treat heart-rate and wearable data as fitness/recovery observations only. Do not diagnose medical conditions or make alarmist claims. If data is missing, say that it was not logged/synced rather than guessing.
              - Do not recommend crash dieting, dehydration, unsafe training volume, or compensating for food with exercise.
              - End with 3-5 specific, practical actions for the next period.
              - Keep the report readable on a phone and under $maxWords words.

              TRACKING DATA:
              ${data.toString()}
            """.trimIndent()
            val text = generateText_(prompt)
            sendResult(
                JSONObject()
                    .put("kind", "period_report")
                    .put("request_id", requestId)
                    .put("report_type", reportType)
                    .put("label", label)
                    .put("start_date", startDate)
                    .put("end_date", endDate)
                    .put("auto", auto)
                    .put("text", text)
            )
        } catch (e: Exception) {
            sendResult(JSONObject().put("kind", "report_error").put("message", "Gemini report failed: ${e.message}"))
        }
    }

'''
marker = '    private suspend fun researchExercise_(name: String) {'
if 'private suspend fun generateReport_' not in s:
    if marker not in s:
        raise SystemExit('Could not locate native report patch point')
    s = s.replace(marker, method + marker)
p.write_text(s)

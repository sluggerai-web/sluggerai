from pathlib import Path
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('fitness-android')

# Version bump over v1.1.3.
p = root / 'app/build.gradle'
s = p.read_text()
s = s.replace('versionCode 7', 'versionCode 8')
s = s.replace("versionName '1.1.3'", "versionName '1.1.4'")
p.write_text(s)

# IMPORTANT: Calendar dates must use the phone's LOCAL timezone, not UTC.
# The previous implementation used new Date().toISOString().slice(0,10),
# which flips to the next day at 8:00 PM during Eastern Daylight Time.
p = root / 'app/src/main/assets/index.html'
s = p.read_text()

old = "function today(){return new Date().toISOString().slice(0,10)}\nfunction dateDaysAgo(n){let d=new Date();d.setDate(d.getDate()-n);return d.toISOString().slice(0,10)}"
new = "function localISODate(d=new Date()){let y=d.getFullYear(),m=String(d.getMonth()+1).padStart(2,'0'),day=String(d.getDate()).padStart(2,'0');return `${y}-${m}-${day}`}\nfunction today(){return localISODate(new Date())}\nfunction dateDaysAgo(n){let d=new Date();d.setDate(d.getDate()-n);return localISODate(d)}"
if old not in s and 'function localISODate' not in s:
    raise SystemExit('Could not locate today/dateDaysAgo timezone patch point')
s = s.replace(old, new)

s = s.replace("let s=start.toISOString().slice(0,10);", "let s=localISODate(start);")
s = s.replace("out.push(d.toISOString().slice(0,10));", "out.push(localISODate(d));")
p.write_text(s)

# Daily history navigation must also stay on local calendar dates.
p = root / 'app/src/main/assets/addon.js'
s = p.read_text()
s = s.replace("let v=d.toISOString().slice(0,10);if(v>today())v=today();", "let v=localISODate(d);if(v>today())v=today();")
p.write_text(s)

# Gemini report periods must use local calendar dates as well.
p = root / 'app/src/main/assets/reportaddon.js'
s = p.read_text()
s = s.replace("function isoDate(d){return d.toISOString().slice(0,10)}", "function isoDate(d){return localISODate(d)}")
p.write_text(s)

# Sanity checks: date-only logic should no longer derive a calendar date from UTC.
idx = (root / 'app/src/main/assets/index.html').read_text()
addon = (root / 'app/src/main/assets/addon.js').read_text()
report = (root / 'app/src/main/assets/reportaddon.js').read_text()
if 'function localISODate' not in idx:
    raise SystemExit('localISODate helper missing after patch')
if "function today(){return localISODate(new Date())}" not in idx:
    raise SystemExit('today() is not local-time based')
if "function isoDate(d){return localISODate(d)}" not in report:
    raise SystemExit('Gemini report calendar dates are not local-time based')
if "let v=localISODate(d);if(v>today())v=today();" not in addon:
    raise SystemExit('Daily activity date navigation is not local-time based')

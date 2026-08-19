from pathlib import Path
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('fitness-android')

# Version bump over v1.1.4.
p = root / 'app/build.gradle'
s = p.read_text()
s = s.replace('versionCode 8', 'versionCode 9')
s = s.replace("versionName '1.1.4'", "versionName '1.1.5'")
p.write_text(s)

# Load retroactive history editor after the Health and Gemini report add-ons.
p = root / 'app/src/main/assets/index.html'
s = p.read_text()
if '<script src="historyaddon.js"></script>' not in s:
    s = s.replace('<script src="reportaddon.js"></script></body></html>', '<script src="reportaddon.js"></script><script src="historyaddon.js"></script></body></html>')
p.write_text(s)

# Sanity checks.
idx = p.read_text()
if '<script src="historyaddon.js"></script>' not in idx:
    raise SystemExit('historyaddon.js was not added to index.html')
b = (root / 'app/build.gradle').read_text()
if 'versionCode 9' not in b or "versionName '1.1.5'" not in b:
    raise SystemExit('v1.1.5 version bump failed')

"""Phase 0: inject data into the web template, producing one self-contained
HTML file that runs from a double-click, no server needed."""

import base64
import json

tpl = open("web_template.html").read()
params = base64.b64encode(open("web_params.bin", "rb").read()).decode()
meta = base64.b64encode(open("web_meta.bin", "rb").read()).decode()
manifest = json.dumps(json.load(open("web_manifest.json")))

out = (tpl.replace("__MANIFEST_JSON__", manifest)
          .replace("__PARAMS_B64__", params)
          .replace("__META_B64__", meta))
open("orrery_phase0.html", "w").write(out)
print(f"orrery_phase0.html: {len(out)/1e6:.1f} MB")

"""Tests for the embedded bytebeat player tabs in app.py.

Executes the real ``_render_bytebeat_player`` source from app.py with a
stubbed ``streamlit`` module, then checks:
1. The main tab copy and the test-tab copy render with unique element IDs.
2. The generated player JavaScript is syntactically valid (node --check).
3. The built-in preset functions evaluate to finite, nonzero audio in node.
"""
import ast
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


class _Ctx:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


class _StreamlitStub:
    def __init__(self):
        self.session_state = {}
        self.captured_html = []

    def selectbox(self, label, options, key=None):
        return options[0]

    def text_area(self, label, value="", height=None, key=None):
        return value

    def slider(self, label, min_value=None, max_value=None, value=None, step=None, key=None):
        return value

    def columns(self, spec):
        n = spec if isinstance(spec, int) else len(spec)
        return [_Ctx() for _ in range(n)]

    def download_button(self, *args, **kwargs):
        return False

    def caption(self, *args, **kwargs):
        return None

    def markdown(self, *args, **kwargs):
        return None


def _load_player_function():
    source = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {"SVG_ICONS", "svg_icon", "svg_dot", "_render_bytebeat_player"}
    parts = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in wanted for t in n.targets
        ):
            parts.append(ast.get_source_segment(source, n))
        elif isinstance(n, ast.FunctionDef) and n.name in wanted:
            parts.append(ast.get_source_segment(source, n))
    return "\n\n".join(parts)


def _render_copy(suffix):
    st_stub = _StreamlitStub()
    components_mod = types.ModuleType("streamlit.components.v1")
    components_mod.html = lambda html, **kw: st_stub.captured_html.append(html)
    pkg = types.ModuleType("streamlit")
    comp_pkg = types.ModuleType("streamlit.components")
    old = {k: sys.modules.get(k) for k in ("streamlit", "streamlit.components", "streamlit.components.v1")}
    sys.modules["streamlit"] = pkg
    sys.modules["streamlit.components"] = comp_pkg
    sys.modules["streamlit.components.v1"] = components_mod
    try:
        namespace = {"st": st_stub, "Path": Path, "json": __import__("json")}
        exec(_load_player_function(), namespace)
        namespace["_render_bytebeat_player"](suffix=suffix)
    finally:
        for k, v in old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return st_stub.captured_html[-1]


def _script_of(html):
    start = html.index("<script>") + len("<script>")
    end = html.index("</script>")
    return html[start:end]


class TestBytebeatPlayerTabs(unittest.TestCase):
    def test_copies_have_unique_element_ids(self):
        main_html = _render_copy("")
        copy_html = _render_copy("_copy")
        self.assertIn("id='bb_play'", main_html)
        self.assertIn("id='bb_play_copy'", copy_html)
        self.assertNotIn("id='bb_play'", copy_html.replace("id='bb_play_copy'", ""))
        self.assertNotIn("bb_play_copy", main_html)

    def test_player_js_syntax_valid(self):
        if shutil.which("node") is None:
            self.skipTest("node not available")
        for suffix in ("", "_copy"):
            js = _script_of(_render_copy(suffix))
            with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
                f.write(js)
                path = f.name
            res = subprocess.run(["node", "--check", path], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"JS syntax error (suffix={suffix!r}): {res.stderr}")

    def test_preset_functions_evaluate(self):
        if shutil.which("node") is None:
            self.skipTest("node not available")
        source = APP_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        consts = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                if n.targets[0].id in ("BB_VIDEO_EQ", "BB_PHONK"):
                    consts[n.targets[0].id] = ast.literal_eval(n.value)
        self.assertIn("BB_VIDEO_EQ", consts)
        self.assertIn("BB_PHONK", consts)
        harness = (
            "const fns = %s;\n"
            "for (const [name, code] of Object.entries(fns)) {\n"
            "  const fn = new Function('t', 'return (' + code + ')');\n"
            "  let mx = 0;\n"
            "  for (let t = 0; t < 8000 * 8; t += 131) {\n"
            "    const v = fn(t);\n"
            "    if (typeof v !== 'number' || !isFinite(v)) { console.log('BAD ' + name); process.exit(1); }\n"
            "    mx = Math.max(mx, Math.abs(v));\n"
            "  }\n"
            "  if (mx < 1e-4) { console.log('SILENT ' + name); process.exit(1); }\n"
            "}\n"
            "console.log('PRESETS OK');\n"
        ) % __import__("json").dumps(consts)
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(harness)
            path = f.name
        res = subprocess.run(["node", path], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertIn("PRESETS OK", res.stdout)

    def test_site_app_js_syntax_and_guards(self):
        site_app = Path(__file__).resolve().parent.parent / "site" / "app.js"
        self.assertTrue(site_app.exists(), "site/app.js must exist")
        code = site_app.read_text(encoding="utf-8")

        # 1. Syntax check with node if available
        if shutil.which("node") is not None:
            res = subprocess.run(["node", "--check", str(site_app)], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"site/app.js syntax error: {res.stderr}")

        # 2. Verify stopPlayback() guards canvas clear with active-playback check
        self.assertIn("if (wasPlaying)", code)
        stop_fn = code[code.find("function stopPlayback()"):code.find("function drawScope()")]
        self.assertIn("if (wasPlaying) {", stop_fn)
        self.assertIn("clearRect", stop_fn)
        # Ensure clearRect is strictly inside the wasPlaying block
        guard_idx = stop_fn.find("if (wasPlaying) {")
        clear_idx = stop_fn.find("clearRect")
        self.assertGreater(clear_idx, guard_idx)

        # 3. Verify Mode change handler mirrors rate change: stops, recompiles, and resumes on playback
        mode_handler = code[code.find("$('mode').onchange ="):code.find("$('rate').onchange =")]
        self.assertIn("if (wasPlaying) stopPlayback()", mode_handler)
        self.assertIn("compile()", mode_handler)
        self.assertIn("compiled = false", mode_handler)


if __name__ == "__main__":
    unittest.main()


# SPDX-License-Identifier: MIT
"""Structural checks for the privacy and no-network rules that need no model to verify."""

import ast
import os
import subprocess
import sys
import unittest

import _sim_bootstrap  # noqa: F401  (must come before moshi imports)

PACKAGE = _sim_bootstrap.PACKAGE_DIR
HEAVY = ("torch", "sphn", "sentencepiece", "huggingface_hub", "gradio")


def read(name):
    with open(os.path.join(PACKAGE, name), encoding="utf-8") as handle:
        return handle.read()


def top_level_imports(source):
    names = []
    for node in ast.parse(source).body:  # module level only: lazy imports inside functions are allowed
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


class SourceRules(unittest.TestCase):
    def test_torch_free_modules_do_not_import_heavy_packages(self):
        for name in ("sim_protocol.py", "sim_session.py", "sim_cli.py", "sim_audio.py", "sim_voices.py"):
            with self.subTest(module=name):
                for module in top_level_imports(read(name)):
                    root = module.split(".")[0]
                    self.assertNotIn(root, HEAVY, f"{name} imports {module}")

    def test_server_has_no_top_level_hugging_face_import(self):
        # Local paths must never reach huggingface_hub; the lazy import only runs without a path.
        for module in top_level_imports(read("server.py")):
            self.assertNotEqual(module.split(".")[0], "huggingface_hub")

    def test_download_counter_call_is_gone(self):
        for name in ("server.py", "offline.py"):
            with self.subTest(module=name):
                self.assertNotIn('"config.json"', read(name))

    def test_gradio_mkcert_and_static_downloads_are_gone(self):
        server = read("server.py")
        self.assertNotIn("from gradio", server)
        self.assertNotIn("import gradio", server)
        self.assertNotIn("ensure_mkcert", server)
        self.assertNotIn("utils.connection", server)
        self.assertNotIn("dist.tgz", server)
        self.assertNotIn("get_lan_ip", server)

    def test_warmup_runs_one_system_prompt_pass_before_the_first_call(self):
        server = read("server.py")
        self.assertIn("state.warmup(voice_prompt=args.default_voice, text_prompt=default_text_prompt)", server)
        self.assertIn("def _warm_system_prompts(", server)

    def test_request_lines_are_never_access_logged(self):
        self.assertIn("access_log=None", read("server.py"))


class ImportPurity(unittest.TestCase):
    def test_session_layer_imports_without_torch_or_model_packages(self):
        code = (
            "import sys, _sim_bootstrap\n"
            "import moshi.sim_protocol, moshi.sim_session, moshi.sim_cli\n"
            "bad = [name for name in %r if name in sys.modules]\n"
            "print(','.join(bad))\n"
            "sys.exit(1 if bad else 0)\n" % (HEAVY,)
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=_sim_bootstrap.TESTS_DIR,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_model_files_compile(self):
        for name in ("server.py", "offline.py", "sim_protocol.py", "sim_session.py", "sim_cli.py"):
            with self.subTest(module=name):
                compile(read(name), name, "exec")
        with open(os.path.join(_sim_bootstrap.REPO_ROOT, "scripts", "sim_call_test.py"), encoding="utf-8") as handle:
            compile(handle.read(), "sim_call_test.py", "exec")


if __name__ == "__main__":
    unittest.main()

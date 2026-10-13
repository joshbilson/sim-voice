# SPDX-License-Identifier: MIT
"""Custom voice ids, path resolution and the on-disk store (moshi.sim_voices). No torch, no GPU."""

import os
import stat
import tempfile
import unittest

import _sim_bootstrap  # noqa: F401  (must come before moshi imports)
from moshi import sim_protocol as P
from moshi import sim_voices as V


class IdTests(unittest.TestCase):
    def test_custom_ids_follow_the_slug_rule(self):
        for good in ("c-ab", "c-warm-sim-k3x9", "c-0-a", "c-" + "a" * 41):
            with self.subTest(good=good):
                self.assertIsNotNone(V.custom_slug(good))
                self.assertTrue(P.CUSTOM_VOICE_RE.fullmatch(good + ".pt"))
        self.assertEqual(V.custom_slug("c-warm"), "warm")

    def test_bad_custom_ids_are_refused(self):
        bad = ["c-", "c-a", "c-A", "c--ab", "c-a_b", "c-a.b", "c-../x", "c-a/b", "c-" + "a" * 42,
               "c-ab\n", " c-ab", "c-ab ", "C-ab", "NATF2", "c-é", "", "c-.."]
        for text in bad:
            with self.subTest(text=text):
                self.assertIsNone(V.custom_slug(text))
                self.assertFalse(P.CUSTOM_VOICE_RE.fullmatch(text + ".pt"))

    def test_preset_names_never_look_custom_and_custom_names_are_not_presets(self):
        self.assertTrue(P.is_preset_name("NATF2.pt"))
        self.assertTrue(P.is_preset_name("VARM4.pt"))
        self.assertFalse(P.is_preset_name("c-ab.pt"))
        self.assertFalse(P.is_preset_name("NATF2.wav"))
        self.assertFalse(P.is_preset_name("../NATF2.pt"))
        self.assertTrue(P.is_voice_name("c-warm-sim-k3x9.pt"))
        self.assertFalse(P.is_voice_name("c-Warm.pt"))
        self.assertFalse(P.is_voice_name("c-" + "a" * 41 + ".pt.pt"))


class ResolutionTests(unittest.TestCase):
    def test_presets_resolve_only_in_the_preset_directory(self):
        self.assertEqual(V.voice_file_path("NATF2.pt", "/presets", "/custom"), os.path.join("/presets", "NATF2.pt"))
        self.assertEqual(V.voice_file_path("NATF2.pt", "/presets", None), os.path.join("/presets", "NATF2.pt"))

    def test_custom_names_resolve_only_in_the_custom_directory(self):
        self.assertEqual(V.voice_file_path("c-ab.pt", "/presets", "/custom"), os.path.join("/custom", "c-ab.pt"))
        with self.assertRaises(ValueError):
            V.voice_file_path("c-ab.pt", "/presets", None)

    def test_unresolvable_names_raise(self):
        for name in ("../NATF2.pt", "c-x.pt", "c-Ab.pt", "NATF2.wav", "/etc/passwd.pt", "c-ab.pt/../x.pt"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                V.voice_file_path(name, "/presets", "/custom")


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.dir = os.path.join(self.root.name, "voices-custom")

    def write(self, name, data=b"x"):
        path = os.path.join(self.dir, name)
        with open(path, "wb") as handle:
            handle.write(data)
        return path

    def test_missing_directory_is_created_private(self):
        V.VoiceStore(self.dir)
        self.assertEqual(stat.S_IMODE(os.stat(self.dir).st_mode), 0o700)

    def test_open_directory_or_file_or_missing_parent_is_refused(self):
        os.mkdir(self.dir, 0o755)
        with self.assertRaises(ValueError) as ctx:
            V.VoiceStore(self.dir)
        self.assertIn("group or world", str(ctx.exception))
        os.chmod(self.dir, 0o700)
        self.assertIsNone(V.check_custom_voice_dir(self.dir))
        file_path = os.path.join(self.root.name, "plain")
        open(file_path, "w").close()
        self.assertIn("not a directory", V.check_custom_voice_dir(file_path))
        self.assertIn("parent", V.check_custom_voice_dir(os.path.join(self.root.name, "no", "such", "dir")))

    def test_names_list_only_regular_custom_pt_files(self):
        store = V.VoiceStore(self.dir)
        self.write("c-alpha.pt")
        self.write("c-alpha.wav")
        self.write("c-beta.pt")
        self.write("NATF2.pt")
        self.write("c-Gamma.pt")
        os.mkdir(os.path.join(self.dir, "c-dir.pt"))
        os.symlink(os.path.join(self.dir, "c-beta.pt"), os.path.join(self.dir, "c-link.pt"))
        self.assertEqual(store.names(), ["c-alpha.pt", "c-beta.pt"])

    def test_install_moves_both_files_privately_and_replaces_atomically(self):
        store = V.VoiceStore(self.dir)
        build = store.new_build_dir()
        self.assertEqual(stat.S_IMODE(os.stat(build).st_mode), 0o700)
        with open(os.path.join(build, "c-warm.pt"), "wb") as handle:
            handle.write(b"first")
        with open(os.path.join(build, "c-warm.wav"), "wb") as handle:
            handle.write(b"RIFF-first")
        store.install(build, "warm")
        self.assertEqual(store.names(), ["c-warm.pt"])
        self.assertTrue(store.has_sample("warm"))
        self.assertEqual(stat.S_IMODE(os.stat(store.pt_path("warm")).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(store.wav_path("warm")).st_mode), 0o600)
        self.assertFalse(os.path.exists(os.path.join(build, "c-warm.pt")))  # moved, not copied
        build = store.new_build_dir()
        with open(os.path.join(build, "c-warm.pt"), "wb") as handle:
            handle.write(b"second")
        with open(os.path.join(build, "c-warm.wav"), "wb") as handle:
            handle.write(b"RIFF-second")
        store.install(build, "warm")
        with open(store.pt_path("warm"), "rb") as handle:
            self.assertEqual(handle.read(), b"second")

    def test_incomplete_build_is_refused_and_installs_nothing(self):
        store = V.VoiceStore(self.dir)
        build = store.new_build_dir()
        with open(os.path.join(build, "c-warm.pt"), "wb") as handle:
            handle.write(b"only the prompt")
        with self.assertRaises(ValueError):
            store.install(build, "warm")
        self.assertEqual(store.names(), [])
        self.assertFalse(store.has_sample("warm"))

    def test_delete_removes_both_files_and_reports_what_existed(self):
        store = V.VoiceStore(self.dir)
        self.write("c-warm.pt")
        self.write("c-warm.wav")
        self.assertTrue(store.delete("warm"))
        self.assertEqual(store.names(), [])
        self.assertFalse(store.delete("warm"))

    def test_stale_build_directories_are_removed_when_the_store_opens(self):
        os.mkdir(self.dir, 0o700)
        stale = os.path.join(self.dir, ".build-leftover")
        os.mkdir(stale, 0o700)
        with open(os.path.join(stale, "c-x.pt"), "wb") as handle:
            handle.write(b"partial")
        V.VoiceStore(self.dir)
        self.assertFalse(os.path.exists(stale))


if __name__ == "__main__":
    unittest.main()

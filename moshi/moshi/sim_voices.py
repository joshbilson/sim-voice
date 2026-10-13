# SPDX-License-Identifier: MIT
#
# Sim fork additions (joshbilson/sim-voice). Licensed like the rest of this repository
# (see LICENSE-MIT and moshi/LICENSE.moshi).
#
# Custom voices on disk: ids, paths, the private directory and atomic installs. Torch-free.
# A custom voice has the id c-<slug> and two files in the custom voice directory:
#   c-<slug>.pt   the PersonaPlex voice prompt (what a call loads)
#   c-<slug>.wav  the 4.0 s, 24 kHz sample it was built from (GET /api/voices/c-<slug>.wav)
# Preset names resolve only in the preset directory, and c- names only in the custom directory.
# Every name is validated by regex before it touches the file system, so no name can escape
# its directory.

from __future__ import annotations

import os
import re
import shutil
import stat
import tempfile
from typing import Optional

from .sim_protocol import CUSTOM_VOICE_RE, is_preset_name

SLUG = r"[a-z0-9][a-z0-9-]{1,40}"
CUSTOM_ID_RE = re.compile(r"c-" + SLUG)
BUILD_PREFIX = ".build-"


def custom_slug(voice_id: str) -> Optional[str]:
    """The slug of a custom id ('c-warm' gives 'warm'), or None when the text is not a custom id."""
    if CUSTOM_ID_RE.fullmatch(voice_id):
        return voice_id[2:]
    return None


def voice_file_path(name: str, preset_dir: Optional[str], custom_dir: Optional[str]) -> str:
    """Where a voice_prompt file name lives. Raises ValueError for a name that cannot resolve."""
    if CUSTOM_VOICE_RE.fullmatch(name):
        if custom_dir is None:
            raise ValueError("custom voices are not enabled on this server")
        return os.path.join(custom_dir, name)
    if is_preset_name(name):
        if preset_dir is None:
            raise ValueError("no voice prompt directory is configured")
        return os.path.join(preset_dir, name)
    raise ValueError("voice name must look like NATF2.pt or c-<slug>.pt")


def check_custom_voice_dir(path: str) -> Optional[str]:
    """None when the directory can be used (a missing one is created 0700). Otherwise the reason."""
    if os.path.lexists(path):
        if not os.path.isdir(path):
            return "--custom-voice-dir exists and is not a directory"
        if os.stat(path).st_mode & 0o077:
            return "--custom-voice-dir must not be group or world accessible (chmod 700)"
        return None
    if not os.path.isdir(os.path.dirname(os.path.abspath(path))):
        return "--custom-voice-dir: its parent directory does not exist"
    return None


def _is_regular(path: str) -> bool:
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


class VoiceStore:
    """The custom voice directory: names, paths, staged builds, atomic install and delete."""

    def __init__(self, directory: str) -> None:
        self.directory = os.path.abspath(directory)
        problem = check_custom_voice_dir(self.directory)
        if problem is not None:
            raise ValueError(problem)
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory, 0o700)
        self._remove_stale_builds()

    def names(self) -> list[str]:
        """Custom voice file names (c-<slug>.pt), sorted. Symlinks and other files are not voices."""
        try:
            entries = os.listdir(self.directory)
        except OSError:
            return []
        return sorted(
            entry for entry in entries
            if CUSTOM_VOICE_RE.fullmatch(entry) and _is_regular(os.path.join(self.directory, entry))
        )

    def pt_path(self, slug: str) -> str:
        return os.path.join(self.directory, f"c-{slug}.pt")

    def wav_path(self, slug: str) -> str:
        return os.path.join(self.directory, f"c-{slug}.wav")

    def has_sample(self, slug: str) -> bool:
        return _is_regular(self.wav_path(slug))

    def new_build_dir(self) -> str:
        """A private staging directory inside the custom directory, so the install is a rename."""
        return tempfile.mkdtemp(prefix=BUILD_PREFIX, dir=self.directory)

    def install(self, build_dir: str, slug: str) -> None:
        """Move a finished build into place. The .pt moves first, because calls load it."""
        pt_src = os.path.join(build_dir, f"c-{slug}.pt")
        wav_src = os.path.join(build_dir, f"c-{slug}.wav")
        for path in (pt_src, wav_src):
            if not _is_regular(path):
                raise ValueError("the build did not produce both voice files")
            os.chmod(path, 0o600)
        os.replace(pt_src, self.pt_path(slug))
        os.replace(wav_src, self.wav_path(slug))

    def delete(self, slug: str) -> bool:
        """Remove both files. True when at least one existed."""
        removed = False
        for path in (self.pt_path(slug), self.wav_path(slug)):
            try:
                os.unlink(path)
                removed = True
            except FileNotFoundError:
                pass
        return removed

    def _remove_stale_builds(self) -> None:
        """Staging directories left by a server that died mid-build."""
        for entry in os.listdir(self.directory):
            path = os.path.join(self.directory, entry)
            if entry.startswith(BUILD_PREFIX) and stat.S_ISDIR(os.lstat(path).st_mode):
                shutil.rmtree(path, ignore_errors=True)

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Serve the Sim protocol with a fake engine: no GPU, no weights. For smoke tests by hand.

The fake engine echoes input audio back and emits the text piece 'a' for every frame.

    python3 moshi/tests/run_fake_server.py --port 8998 --auth-token-file /path/to/token
    python3 scripts/sim_call_test.py --url http://127.0.0.1:8998 --token-file /path/to/token --seconds 2
"""

import sys

from aiohttp import web

import _sim_bootstrap  # noqa: F401  (must come before moshi imports)
from fake_engine import FakeEngine
from moshi.sim_cli import build_arg_parser, validate_serve_args
from moshi.sim_session import SessionConfig, create_app
from moshi.utils.logging import setup_logger


def main(argv=None) -> None:
    setup_logger("moshi.sim_session")  # the same handler server.py installs
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    token, default_text = validate_serve_args(parser, args)
    engine = FakeEngine(custom_voice_dir=args.custom_voice_dir)
    if args.default_voice not in engine.voices():
        parser.error("the fake engine offers only NATF0.pt and NATF2.pt")
    config = SessionConfig(
        default_voice=args.default_voice,
        default_text_prompt=default_text,
        auth_token=token,
        max_backlog_frames=args.max_backlog_frames,
        idle_timeout_s=args.idle_timeout_s,
        max_session_s=args.max_session_seconds,
        stats_interval_s=args.stats_interval_s,
        custom_voice_dir=args.custom_voice_dir,
    )
    print("fake engine: echoes audio, emits 'a' per frame, loads no model", file=sys.stderr)
    web.run_app(create_app(engine, config), host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()

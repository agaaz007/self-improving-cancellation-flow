from __future__ import annotations

import argparse
import errno
import io
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from cta_autoresearch import lab_cli


class LabCliTest(unittest.TestCase):
    def test_run_manager_tracks_progress_and_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = lab_cli._make_run_manager(Path(tmpdir), {"population": 20, "strategy_depth": "quick"})

            def fake_dataset(*, progress_callback=None, **_settings):
                if progress_callback:
                    progress_callback(0.25, "settings", "Loaded settings.")
                    progress_callback(0.6, "validation", "Validated shortlist.")
                return {
                    "meta": {
                        "top_strategy": "Example winner",
                        "model_backend": "heuristic:heuristic-simulator",
                    },
                    "top_strategies": [],
                    "personas": [],
                    "idea_agents": [],
                    "all_candidates": [],
                    "best_non_discount": [],
                    "dimensions": {},
                    "top_patterns": {},
                    "segment_leaders": {},
                    "controls": {},
                }

            with mock.patch.object(lab_cli, "build_dashboard_dataset", side_effect=fake_dataset):
                created = manager["create_job"]({"validation_budget": 80})
                self.assertIn(created["status"], {"queued", "running", "completed"})

                final = None
                for _ in range(40):
                    final = manager["get_job"](created["id"])
                    if final and final["status"] == "completed":
                        break
                    time.sleep(0.01)

            self.assertIsNotNone(final)
            self.assertEqual(final["status"], "completed")
            self.assertEqual(final["result"]["meta"]["top_strategy"], "Example winner")
            self.assertEqual(final["result_meta"]["model_backend"], "heuristic:heuristic-simulator")
            self.assertGreaterEqual(len(final["activity_log"]), 3)
            self.assertIn("run-sample", final["command_preview"])

    def test_serve_dashboard_skips_snapshot_build_before_binding(self) -> None:
        fake_parser = mock.Mock()
        fake_server = mock.Mock()
        fake_server.__enter__ = mock.Mock(return_value=fake_server)
        fake_server.__exit__ = mock.Mock(return_value=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                command="serve-dashboard",
                population=12,
                top_n=5,
                seed=7,
                strategy_depth="quick",
                persona_richness="rich",
                ideation_agents=3,
                validation_budget=80,
                model_name="gpt-5.4-mini",
                output_dir=tmpdir,
                port=8765,
            )
            fake_parser.parse_args.return_value = args

            with (
                mock.patch.object(lab_cli, "build_parser", return_value=fake_parser),
                mock.patch.object(lab_cli, "write_dashboard_data") as write_dashboard_data,
                mock.patch.object(lab_cli, "_make_handler", return_value=object()) as make_handler,
                mock.patch("http.server.ThreadingHTTPServer", return_value=fake_server) as server_class,
            ):
                lab_cli.main()
                self.assertTrue(Path(tmpdir).exists())

        write_dashboard_data.assert_not_called()
        make_handler.assert_called_once()
        server_class.assert_called_once()
        fake_server.serve_forever.assert_called_once()

    def test_serve_dashboard_reports_port_in_use_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                sys,
                "argv",
                [
                    "lab_cli.py",
                    "serve-dashboard",
                    "--output-dir",
                    tmpdir,
                    "--port",
                    "8000",
                ],
            ):
                with mock.patch("http.server.ThreadingHTTPServer", side_effect=OSError(errno.EADDRINUSE, "Address in use")):
                    stderr = io.StringIO()
                    with mock.patch("sys.stderr", stderr):
                        with self.assertRaises(SystemExit) as ctx:
                            lab_cli.main()

        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("Port 8000 is already in use.", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

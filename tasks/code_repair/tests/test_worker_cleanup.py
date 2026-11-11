"""Real child-process regressions for the model-free scheduling path."""
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evaluation import run_workers


class WorkerCleanup(unittest.TestCase):
    def exercise(self, interrupt=False):
        children = []
        popen = subprocess.Popen
        sleep = time.sleep
        previous = signal.getsignal(signal.SIGTERM)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / 'logs').mkdir()

            def launch(command, **kwargs):
                # One failing worker and one still running, using real processes.
                seed = command[command.index('--seed') + 1]
                body = 'import time; time.sleep(60)' if interrupt or seed != '0' else 'raise SystemExit(7)'
                process = popen([sys.executable, '-c', body], **kwargs)
                children.append(process)
                return process

            def tick(seconds):
                if interrupt:
                    signal.raise_signal(signal.SIGTERM)
                sleep(.01)

            try:
                with patch('evaluation.subprocess.Popen', side_effect=launch), patch('evaluation.time', SimpleNamespace(sleep=tick, time=time.time, monotonic=time.monotonic)):
                    with self.assertRaises(KeyboardInterrupt if interrupt else RuntimeError):
                        run_workers(SimpleNamespace(gpus='0,1'), output, output)
                self.assertEqual(len(children), 2)
                self.assertTrue(all(p.poll() is not None for p in children))
                self.assertIs(signal.getsignal(signal.SIGTERM), previous)
            finally:
                for process in children:
                    if process.poll() is None:
                        process.kill()
                    process.wait()

    def test_failed_shard_is_requeued_with_clean_output(self):
        popen, sleep, calls = subprocess.Popen, time.sleep, []
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / 'logs').mkdir()
            (output / 'public').mkdir()

            def launch(command, **kwargs):
                seed, shard = command[command.index('--seed') + 1], command[command.index('--shard') + 1]
                calls.append((seed, shard))
                first = calls.count((seed, shard)) == 1
                partial = output / 'public' / f'seed{seed}.shard{shard}.jsonl'
                self.assertFalse(partial.exists())
                if first and (seed, shard) == ('1', '2'):
                    partial.write_text('{}\n')  # partial rows from the run that dies
                    return popen([sys.executable, '-c', 'raise SystemExit(1)'], **kwargs)
                return popen([sys.executable, '-c', ''], **kwargs)

            with patch('evaluation.subprocess.Popen', side_effect=launch), \
                 patch('evaluation.time', SimpleNamespace(sleep=lambda s: sleep(.01), time=time.time, monotonic=time.monotonic)):
                run_workers(SimpleNamespace(gpus='0', worker_retries=2, retry_backoff=0), output, output)
            self.assertEqual(len(calls), 25)
            self.assertEqual(calls.count(('1', '2')), 2)
            self.assertTrue((output / 'logs' / 'seed1.shard2.attempt1.log').exists())

    def test_failure_stops_other_workers(self):
        self.exercise()

    def test_sigterm_stops_workers_and_restores_handler(self):
        self.exercise(interrupt=True)


if __name__ == '__main__':
    unittest.main()

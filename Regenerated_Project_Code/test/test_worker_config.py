import os
import unittest

from Regenerated_Project_Code.worker_utils import resolve_num_workers


class WorkerConfigTest(unittest.TestCase):
    def test_forced_zero_workers(self):
        workers, use_workers = resolve_num_workers(0)
        self.assertEqual(workers, 0)
        self.assertFalse(use_workers)

    def test_auto_mode_respects_cpu_count(self):
        workers, use_workers = resolve_num_workers(-1)
        cpu_cnt = os.cpu_count() or 4
        expected = min(8, max(2, cpu_cnt // 2))
        self.assertEqual(workers, expected)
        self.assertEqual(use_workers, workers > 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

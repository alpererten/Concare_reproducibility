import unittest

from Regenerated_Project_Code.worker_utils import resolve_num_workers


class ChooseWorkersTest(unittest.TestCase):
    def test_user_override_zero(self):
        workers, use_workers = resolve_num_workers(0)
        self.assertEqual(workers, 0)
        self.assertFalse(use_workers)

    def test_auto_mode_positive(self):
        workers, use_workers = resolve_num_workers(-1)
        self.assertGreaterEqual(workers, 0)
        if workers == 0:
            self.assertFalse(use_workers)
        else:
            self.assertTrue(use_workers)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

import unittest

from tools.worker_device import (
    NoAcceleratorError,
    resolve_worker_count,
    resolve_worker_device,
)


class ResolveWorkerDeviceTests(unittest.TestCase):
    def test_plain_cuda_fans_out_one_worker_per_gpu(self) -> None:
        self.assertEqual(resolve_worker_device("cuda", 0), "cuda:0")
        self.assertEqual(resolve_worker_device("cuda", 3), "cuda:3")

    def test_shared_devices_ignore_the_worker_index(self) -> None:
        for spec in ("cpu", "mps", "cuda:1"):
            self.assertEqual(resolve_worker_device(spec, 0), spec)
            self.assertEqual(resolve_worker_device(spec, 2), spec)


class ResolveWorkerCountTests(unittest.TestCase):
    def test_cuda_is_capped_at_the_number_of_gpus(self) -> None:
        self.assertEqual(resolve_worker_count(8, "cuda", available_cuda=2), 2)
        self.assertEqual(resolve_worker_count(1, "cuda", available_cuda=2), 1)

    def test_cuda_without_a_gpu_fails_loudly(self) -> None:
        # Silently falling back to CPU would turn a two-minute job into an overnight one
        # on a billing instance, so the caller has to ask for CPU explicitly.
        with self.assertRaises(NoAcceleratorError):
            resolve_worker_count(1, "cuda", available_cuda=0)

    def test_cpu_keeps_the_requested_worker_count(self) -> None:
        self.assertEqual(resolve_worker_count(6, "cpu", available_cuda=0), 6)

    def test_mps_is_forced_to_a_single_worker(self) -> None:
        # Apple Silicon exposes one device and torch has no multi-process story for it;
        # four workers would contend for the same GPU rather than divide the work.
        self.assertEqual(resolve_worker_count(4, "mps", available_cuda=0), 1)

    def test_an_explicit_cuda_index_is_a_single_worker(self) -> None:
        self.assertEqual(resolve_worker_count(4, "cuda:1", available_cuda=4), 1)

    def test_a_worker_count_below_one_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_worker_count(0, "cpu", available_cuda=0)


if __name__ == "__main__":
    unittest.main()

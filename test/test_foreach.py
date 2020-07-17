import collections
import unittest

import torch
import torch.cuda
from test_torch import AbstractTestCases
from torch.testing._internal.common_utils import TestCase, run_tests

class TestCudaComm(TestCase):
    def test_foreach_tensor_add_scalar(self):
        N = 2
        H = 2
        W = 2

        tensors = []
        for dt in torch.testing.get_all_dtypes():
            for d in torch.testing.get_all_device_types():
                for _ in range(N):
                    tensors.append(torch.zeros(H, W, device=d, dtype=dt))

                res = torch.foreach_add(tensors, 1)
                print(tensors)
                print(res)
                print("\n")

                for t in res: 
                    print(dt)
                    print(t.dtype)
                    print(torch.ones(H, W, device=d, dtype=dt).dtype)
                    self.assertEqual(t, torch.ones(H, W, device=d, dtype=dt))

if __name__ == '__main__':
    run_tests()
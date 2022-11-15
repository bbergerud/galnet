import torch
import unittest
from galnet.layers.padding import *

class TestPadding(unittest.TestCase):
    def test_wrap_pad_2d(self):
        x = torch.arange(9).reshape(1,1,3,3)
        self.assertTrue(torch.equal(
            wrap_pad_2d(x, padding=1, dim=0, pad_both=True),
            torch.tensor([[[[0, 6, 7, 8, 0],
                            [0, 0, 1, 2, 0],
                            [0, 3, 4, 5, 0],
                            [0, 6, 7, 8, 0],
                            [0, 0, 1, 2, 0]]]])))
        self.assertTrue(torch.equal(
            wrap_pad_2d(x, padding=1, dim=0, pad_both=False),
            torch.tensor([[[[6, 7, 8],
                            [0, 1, 2],
                            [3, 4, 5],
                            [6, 7, 8],
                            [0, 1, 2]]]])))    
        self.assertTrue(torch.equal(
            wrap_pad_2d(x, padding=1, dim=1, pad_both=True),
            torch.tensor([[[[0, 0, 0, 0, 0],
                            [2, 0, 1, 2, 0],
                            [5, 3, 4, 5, 3],
                            [8, 6, 7, 8, 6],
                            [0, 0, 0, 0, 0]]]])))
        self.assertTrue(torch.equal(
            wrap_pad_2d(x, padding=1, dim=1, pad_both=False),
            torch.tensor([[[[2, 0, 1, 2, 0],
                            [5, 3, 4, 5, 3],
                            [8, 6, 7, 8, 6]]]])))

    def test_WrapPad2d(self):
        x = torch.arange(9).reshape(1,1,3,3)
        self.assertTrue(torch.equal(
            WrapPad2d(padding=2, dim=0)(x),
            torch.tensor([[[[0, 0, 3, 4, 5, 0, 0],
                            [0, 0, 6, 7, 8, 0, 0],
                            [0, 0, 0, 1, 2, 0, 0],
                            [0, 0, 3, 4, 5, 0, 0],
                            [0, 0, 6, 7, 8, 0, 0],
                            [0, 0, 0, 1, 2, 0, 0],
                            [0, 0, 3, 4, 5, 0, 0]]]])))
        self.assertTrue(torch.equal(
            WrapPad2d(padding=2, dim=1)(x),
            torch.tensor([[[[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [1, 2, 0, 1, 2, 0, 1],
                            [4, 5, 3, 4, 5, 3, 4],
                            [7, 8, 6, 7, 8, 6, 7],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]]]])))

if __name__ == '__main__':
    unittest.main(exit=False)
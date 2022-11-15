import unittest
import torch
import torch.nn as nn
import galnet.utils as utils

class TestUtils(unittest.TestCase):
    def test_count_parameters(self):
        model = nn.Linear(10,10, bias=False)
        self.assertEqual(utils.count_parameters(model), 100)

    def test_get_merged_channels(self):
        # Test that add, mul, cat, and concat work properly
        channels = [5,5,5,5]
        self.assertEqual(5, utils.get_merged_channels(channels, merge_operation='add'))
        self.assertEqual(5, utils.get_merged_channels(channels, merge_operation='mul'))
        self.assertEqual(20, utils.get_merged_channels(channels, merge_operation='cat'))
        self.assertEqual(20, utils.get_merged_channels(channels, merge_operation='concat'))
        # Test exception cases for add and mul
        with self.assertRaises(ValueError):
            for op in ['add', 'mul']:
                utils.get_merged_channels([5,3],merge_operation=op)
        with self.assertRaises(NotImplementedError):
            utils.get_merged_channels([5,3], merge_operation='N/A')

    def test_get_padding(self):
        """Tests that padding results in the same spatial size after convolution"""
        for dim in [20,21]: # Test odd and even sizes
            x = torch.randn(1,1,dim,dim)
            for k in [3,5,7]:   # Test odd kernel sizes
                for d in [1,2,3]:   # Test dilation
                    y = torch.nn.Conv2d(
                        in_channels  = 1,
                        out_channels = 1,
                        kernel_size  = k,
                        dilation = d,
                        padding  = utils.get_padding(k,d)
                    )(x)
                    self.assertEqual(x.shape, y.shape)

if __name__ == '__main__':
    unittest.main(exit=False)
import unittest

import torch

from c2l.models.loftr.resnet_fpn import ResNetFPN


class TestResNetFPN(unittest.TestCase):

    def test_forward(self):
        x = torch.randn(1, 1, 640, 480)

        # Test the two configs found in the LOFTR repository
        model8_2 = ResNetFPN(input_dim=1, initial_dim=128, block_dims=[128, 196, 256],
                             num_down=2, num_up=2)

        model16_4 = ResNetFPN(input_dim=1, initial_dim=128, block_dims=[128, 196, 256, 512],
                              num_down=3, num_up=2)

        y8, y2 = model8_2(x)
        self.assertEqual(y8.shape, (1, 256, 80, 60))
        self.assertEqual(y2.shape, (1, 128, 320, 240))

        y16, y4 = model16_4(x)
        self.assertEqual(y16.shape, (1, 512, 40, 30))
        self.assertEqual(y4.shape, (1, 196, 160, 120))

import unittest

import torch

from c2l.models.loftr.c2ltregressor1 import FeatureWithMask
from c2l.models.loftr.resnet_fpn import ResNetFPN


class TestResNetFPN(unittest.TestCase):

    def setUp(self):
        # Test the two configs found in the LOFTR repository
        self.model8_2 = ResNetFPN(input_dim=1, initial_dim=128, block_dims=[128, 196, 256],
                                  num_down=2, num_up=2)

        self.model16_4 = ResNetFPN(input_dim=1, initial_dim=128, block_dims=[128, 196, 256, 512],
                                   num_down=3, num_up=2)

    def test_forward(self):
        x = FeatureWithMask(
            feat=torch.randn(1, 1, 640, 480),
            mask=None
        )

        y8, y2 = self.model8_2(x)
        y8, y2 = y8.feat, y2.feat
        self.assertEqual(y8.shape, (1, 256, 80, 60))
        self.assertEqual(y2.shape, (1, 128, 320, 240))

        y16, y4 = self.model16_4(x)
        y16, y4 = y16.feat, y4.feat
        self.assertEqual(y16.shape, (1, 512, 40, 30))
        self.assertEqual(y4.shape, (1, 196, 160, 120))

    def test_forward_with_mask(self):
        N, C, H, W = 2, 1, 640, 480
        x = FeatureWithMask(
            feat=torch.randn(N, C, H, W),
            mask=torch.randn(N, H, W).abs() > 0.5
        )

        y8, y2 = self.model8_2(x)

        self.assertEqual(y8.feat.shape, (N, 256, 80, 60))
        self.assertEqual(y8.mask.shape, (N, 80, 60))
        self.assertEqual(y2.feat.shape, (N, 128, 320, 240))
        self.assertEqual(y2.mask.shape, (N, 320, 240))

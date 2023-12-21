import unittest

import torch

from c2l.models.loftr.loftr import LocalFeatureTransformer
from c2l.models.loftr.visloc1 import FeatureWithMask


class TestLocalFeatureTransformer(unittest.TestCase):

    def setUp(self):
        self.dim = 16
        self.nhead = 4
        self.nlayers = 3
        self.transformer = LocalFeatureTransformer(self.dim, self.nhead, self.nlayers)
        self.batch_size = 2
        self.seq_len_img = 10
        self.seq_len_pcl = 8

    def generate_random_input(self):
        feat_img = FeatureWithMask(
            feat=torch.randn(self.batch_size, self.seq_len_img, self.dim),
            mask=torch.randint(0, 2, (self.batch_size, self.seq_len_img)).float()
        )
        feat_pcl = FeatureWithMask(
            feat=torch.randn(self.batch_size, self.seq_len_pcl, self.dim),
            mask=torch.randint(0, 2, (self.batch_size, self.seq_len_pcl)).float()
        )
        return feat_img, feat_pcl

    def test_forward_pass(self):
        feat_img, feat_pcl = self.generate_random_input()

        out_feat_img, out_feat_pcl = self.transformer(feat_img, feat_pcl)

        # Check output shapes
        self.assertEqual(out_feat_img.feat.shape, (self.batch_size, self.seq_len_img, self.dim))
        self.assertEqual(out_feat_pcl.feat.shape, (self.batch_size, self.seq_len_pcl, self.dim))

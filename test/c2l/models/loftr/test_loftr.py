import unittest

import torch

from c2l.models.loftr.loftr import LocalFeatureTransformer


class TestLocalFeatureTransformer(unittest.TestCase):

    def setUp(self):
        self.dim = 16
        self.nhead = 4
        self.nlayers = 3
        self.transformer = LocalFeatureTransformer(self.dim, self.nhead, self.nlayers)
        self.batch_size = 2
        self.seq_len_img = 10
        self.seq_len_pcl = 8

    def generate_random_tensors(self):
        feat_img = torch.randn(self.batch_size, self.seq_len_img, self.dim)
        feat_pcl = torch.randn(self.batch_size, self.seq_len_pcl, self.dim)
        return feat_img, feat_pcl

    def test_forward_pass(self):
        feat_img, feat_pcl = self.generate_random_tensors()
        mask_img = torch.randint(0, 2, (self.batch_size, self.seq_len_img)).float()
        mask_pcl = torch.randint(0, 2, (self.batch_size, self.seq_len_pcl)).float()

        out_feat_img, out_feat_pcl = self.transformer(feat_img, feat_pcl, mask_img, mask_pcl)

        # Check output shapes
        self.assertEqual(out_feat_img.shape, (self.batch_size, self.seq_len_img, self.dim))
        self.assertEqual(out_feat_pcl.shape, (self.batch_size, self.seq_len_pcl, self.dim))

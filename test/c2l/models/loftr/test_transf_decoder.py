import unittest

import torch

from c2l.models.loftr.c2ltregressor1 import FeatureWithMask
from c2l.models.loftr.transf_decoder import TransformationDecoder


class TestTransformationDecoder(unittest.TestCase):

    def setUp(self):
        self.dim = 256
        nhead = 8
        nlayers = 4
        heads = {
            "trans": torch.nn.Linear(self.dim, 6),
            "rot": torch.nn.Linear(self.dim, 5),
        }
        self.model = TransformationDecoder(self.dim, nhead, nlayers, heads)

    def test_xavier_uniform_initialization(self):
        for p in self.model.parameters():
            if p.dim() > 1:
                # Test that parameters are initialized with Xavier Uniform
                # pylint: disable=protected-access
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(p)
                expected_std = (2.0 / (fan_in + fan_out))**0.5
                bounds = expected_std * 3  # Assuming 99.7% should fall within 3 standard deviations
                self.assertTrue(torch.all(p >= -bounds) and torch.all(p <= bounds))

    def test_output_shape(self):
        N, L, C = 10, 50, self.dim
        img_feat = FeatureWithMask(
            feat=torch.randn(N, L, C),
            mask=torch.randint(0, 2, (N, L)).float()
        )
        pcl_feat = FeatureWithMask(
            feat=torch.randn(N, L, C),
            mask=torch.randint(0, 2, (N, L)).float()
        )

        twu = self.model(img_feat, pcl_feat)

        self.assertEqual(twu.trans.shape, (N, 3))
        self.assertEqual(twu.trans_unc.shape, (N, 3))
        self.assertEqual(twu.rot.shape, (N, 4))
        self.assertEqual(twu.rot_unc.shape, (N, 1))

    def test_unit_quaternion(self):
        N, L, C = 10, 50, self.dim
        img_feat = FeatureWithMask(
            feat=torch.randn(N, L, C),
            mask=torch.randint(0, 2, (N, L)).float()
        )
        pcl_feat = FeatureWithMask(
            feat=torch.randn(N, L, C),
            mask=torch.randint(0, 2, (N, L)).float()
        )

        twu = self.model(img_feat, pcl_feat)
        norms = torch.norm(twu.rot, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))

    def test_with_and_without_masks(self):
        N, L, C = 10, 50, self.dim
        img_feat_ = torch.randn(N, L, C)
        pcl_feat_ = torch.randn(N, L, C)
        img_feat = FeatureWithMask(
            feat=img_feat_,
            mask=None
        )
        pcl_feat = FeatureWithMask(
            feat=pcl_feat_,
            mask=None
        )
        img_feat_with_mask = FeatureWithMask(
            feat=img_feat_,
            mask=torch.ones(N, L)
        )
        pcl_feat_with_mask = FeatureWithMask(
            feat=pcl_feat_,
            mask=torch.ones(N, L)
        )

        twu_masked = self.model(img_feat, pcl_feat)
        twu_no_mask = self.model(img_feat_with_mask, pcl_feat_with_mask)

        self.assertEqual(torch.sum(twu_masked.trans - twu_no_mask.trans), 0)
        self.assertEqual(torch.sum(twu_masked.trans_unc - twu_no_mask.trans_unc), 0)
        self.assertEqual(torch.sum(twu_masked.rot - twu_no_mask.rot), 0)
        self.assertEqual(torch.sum(twu_masked.rot_unc - twu_no_mask.rot_unc), 0)

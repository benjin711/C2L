import unittest

import torch
from omegaconf import OmegaConf

from c2l.utils.builders import build_models


class TestC2LTRegressor1(unittest.TestCase):

    def test_instatiation(self):
        # Instantiation of C2LTRegressor1 with its dedicated config file
        config = OmegaConf.load('c2l/conf/models/c2ltregressor1.yaml')
        models = build_models(config)
        self.assertTrue(models.keys() == {'c2ltregressor1'})

    def test_forward(self):
        # Forward pass of C2LTRegressor1
        config = OmegaConf.load('c2l/conf/models/c2ltregressor1.yaml')
        model = build_models(config)['c2ltregressor1']

        img = torch.rand(1, 3, 640, 480)
        pcl = torch.rand(1, 1000, 4)

        data = {
            'img': img,
            'pcl': pcl,
        }

        # Test forward pass
        model(data)

        # Test output availability
        self.assertTrue("trans" in data)
        self.assertTrue("trans_unc" in data)
        self.assertTrue("rot" in data)
        self.assertTrue("rot_unc" in data)

        # Test that input data is not modified
        self.assertTrue(torch.equal(data['img'], img))
        self.assertTrue(torch.equal(data['pcl'], pcl))

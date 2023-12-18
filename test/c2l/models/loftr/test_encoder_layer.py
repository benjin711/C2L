import unittest

import torch

from c2l.models.loftr.loftr import LoFTREncoderLayer


class TestLoFTREncoderLayer(unittest.TestCase):

    def setUp(self):
        self.orig_dim = 16
        self.nhead = 4
        self.encoder_layer = LoFTREncoderLayer(self.orig_dim, self.nhead)
        self.batch_size = 2
        self.seq_len = 10

    def generate_random_tensors(self):
        x = torch.randn(self.batch_size, self.seq_len, self.orig_dim)
        source = torch.randn(self.batch_size, self.seq_len, self.orig_dim)
        return x, source

    def test_forward_pass(self):
        x, source = self.generate_random_tensors()

        # Optional masks (can be None)
        x_mask = torch.randint(0, 2, (self.batch_size, self.seq_len)).float()
        source_mask = torch.randint(0, 2, (self.batch_size, self.seq_len)).float()

        output = self.encoder_layer(x, source, x_mask, source_mask)

        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.orig_dim)
        self.assertEqual(output.shape, expected_shape)

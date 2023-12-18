import unittest

import torch

from c2l.models.loftr.linear_attention import LinearAttention


class TestLinearAttention(unittest.TestCase):

    def setUp(self):
        self.attention = LinearAttention()
        self.batch_size = 2
        self.seq_len = 10
        self.heads = 3
        self.d_model = 4

    def generate_random_tensors(self):
        queries = torch.randn(self.batch_size, self.seq_len, self.heads, self.d_model)
        keys = torch.randn(self.batch_size, self.seq_len, self.heads, self.d_model)
        values = torch.randn(self.batch_size, self.seq_len, self.heads, self.d_model)
        return queries, keys, values

    def test_forward_pass_without_masks(self):
        queries, keys, values = self.generate_random_tensors()
        output = self.attention(queries, keys, values)

        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.heads, self.d_model)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_pass_with_masks(self):
        queries, keys, values = self.generate_random_tensors()
        q_mask = torch.randint(0, 2, (self.batch_size, self.seq_len))
        kv_mask = torch.randint(0, 2, (self.batch_size, self.seq_len))

        output = self.attention(queries, keys, values, q_mask, kv_mask)

        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.heads, self.d_model)
        self.assertEqual(output.shape, expected_shape)

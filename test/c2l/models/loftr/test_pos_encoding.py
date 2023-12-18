import unittest

import torch

from c2l.models.loftr.pos_encodings import PositionEncodingSine


class TestPositionEncodingSine(unittest.TestCase):

    def test_init(self):
        # Test for different dimensions
        for dim in [64, 128, 256]:
            encoder = PositionEncodingSine(dim)
            self.assertEqual(encoder.pos_encoding.shape, (1, dim, 256, 256))

            # Check the range of values in the encoding
            self.assertTrue(torch.all(encoder.pos_encoding >= -1.0))
            self.assertTrue(torch.all(encoder.pos_encoding <= 1.0))

    def test_forward_pass(self):
        dim = 128
        encoder = PositionEncodingSine(dim)
        input_tensor = torch.randn(1, dim, 100, 100)  # Example input

        # Perform forward pass
        output = encoder(input_tensor)

        # Check output shape
        self.assertEqual(output.shape, input_tensor.shape)

        # Check if encoding is added
        encoding_cropped = encoder.pos_encoding[:, :, :100, :100]
        expected_output = input_tensor + encoding_cropped
        self.assertTrue(torch.allclose(output, expected_output))

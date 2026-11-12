import unittest
import torch
from panovln_dapo.fsdp_batch_probe import gradient_statistics,summarize_gradient_statistics

class GradientDiagnosticsTests(unittest.TestCase):
    def test_scaling_has_same_direction_but_nonzero_relative_error(self):
        result=summarize_gradient_statistics(gradient_statistics(torch.tensor([1.,-2.,0.]),torch.tensor([2.,-4.,0.]),chunk_elements=1))
        self.assertAlmostEqual(result['relative_l2'],1.)
        self.assertAlmostEqual(result['cosine'],1.)
        self.assertEqual(result['sign_disagreements'],0)

    def test_opposite_direction_and_partition_aggregation(self):
        a=gradient_statistics(torch.tensor([1.,2.]),torch.tensor([-1.,-2.]),chunk_elements=1)
        b=gradient_statistics(torch.tensor([3.]),torch.tensor([-3.]))
        result=summarize_gradient_statistics({key:a[key]+b[key] for key in a})
        self.assertAlmostEqual(result['relative_l2'],2.)
        self.assertAlmostEqual(result['cosine'],-1.)
        self.assertEqual(result['sign_disagreements'],3)

    def test_zero_reference_is_explicitly_undefined(self):
        result=summarize_gradient_statistics(gradient_statistics(torch.zeros(2),torch.ones(2)))
        self.assertIsNone(result['relative_l2'])
        self.assertIsNone(result['cosine'])

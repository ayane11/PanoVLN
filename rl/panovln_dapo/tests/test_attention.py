import unittest
from panovln_dapo.attention import resolve_attention

class AttentionTests(unittest.TestCase):
    def test_auto_backend_tracks_compute_dtype(self):
        self.assertEqual(resolve_attention('auto','bf16'),'flash_attention_2')
        self.assertEqual(resolve_attention('auto','fp32'),'sdpa')
    def test_explicit_sdpa_is_available_for_bf16_diagnostics(self):
        self.assertEqual(resolve_attention('sdpa','bf16'),'sdpa')
    def test_unsupported_backend_and_fp32_flash_fail(self):
        for backend,precision in [('flash_attention_2','fp32'),('unknown','bf16'),('auto','fp16')]:
            with self.assertRaises(ValueError):resolve_attention(backend,precision)

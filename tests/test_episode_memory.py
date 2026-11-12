import unittest
import torch
from src.memory import EpisodeMemory, integrate_actions


class EpisodeMemoryTests(unittest.TestCase):
    def make(self, **extra):
        return EpisodeMemory(dict(feature_dim=16, heads=4, tokens=3, bank_frames=3,
            frame_tokens=4, pose_hidden=16, chunk_frames=2, **extra))

    def test_odometry_and_long_time(self):
        x, y, yaw = integrate_actions(['left'] * 24 + ['forward'])
        self.assertAlmostEqual(x, 0)
        self.assertAlmostEqual(y, .25)
        self.assertEqual(yaw, 0)
        m = self.make()
        self.assertFalse(torch.equal(m.time_embedding(torch.tensor([500])), m.time_embedding(torch.tensor([900]))))
        with self.assertRaises(ValueError):
            integrate_actions(['stop'])

    def test_gradients_capacity_and_cache(self):
        torch.manual_seed(3)
        m = self.make()
        raw = torch.randn(12, 16, requires_grad=True)
        for step in range(5):
            m.write('a', step, (0., step * .25, 0.), raw + step)
        self.assertEqual(len(m.banks['a']), 3)
        self.assertFalse(next(iter(m.banks['a'].values()))[1].requires_grad)
        out = m.read('a', 4, (0., 1., 0.))
        out.square().mean().backward()
        self.assertIsNone(raw.grad)
        for component in (m.tome, m.retrieval, m.pos_embed):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in component.parameters()))
        m.eval()
        with torch.no_grad():
            a = m.read('a', 4, (0., 1., 0.))
            b = m.read('a', 4, (0., 1., 0.))
        torch.testing.assert_close(a, b)
        self.assertEqual(len(m.grounded['a']), 3)
        clone = self.make().eval()
        clone.load_state_dict(m.state_dict())
        clone.import_state(m.export_state())
        with torch.no_grad():
            torch.testing.assert_close(a, clone.read('a', 4, (0., 1., 0.)))
        m.reset('a')
        self.assertFalse(m.banks)

    def test_duplicate_after_frme_removed_latest_timestamp(self):
        m = EpisodeMemory(dict(feature_dim=16, heads=4, tokens=3,
            bank_frames=1, frame_tokens=4, pose_hidden=16))
        raw = torch.randn(8, 16)
        m.write('a', 0, (0., 0., 0.), raw)
        m.write('a', 1, (0., .25, 0.), raw + 1)
        self.assertNotIn(1, m.banks['a'])
        merged = m.banks['a'][0][1].clone()
        m.write('a', 1, (0., .25, 0.), raw + 100)
        torch.testing.assert_close(m.banks['a'][0][1], merged)
        state = m.export_state()
        m.import_state(state)
        m.write('a', 1, (0., .25, 0.), raw + 100)
        torch.testing.assert_close(m.banks['a'][0][1], merged)
        target = EpisodeMemory(m.settings).to(dtype=torch.bfloat16)
        target.import_state(state)
        self.assertEqual(target.banks['a'][0][1].dtype, torch.bfloat16)
        target.write('a', 2, (0., .5, 0.), raw.to(torch.bfloat16))
        with self.assertRaises(ValueError):
            m.read('a', 0, (0., 0., 0.))

    def test_sdpa_compression_matches_upstream_attention(self):
        from src.memory.tome.merge import bipartite_soft_matching, merge_wavg
        m = self.make().eval()
        raw = torch.randn(2, 12, 16)
        reference = raw
        for block in m.tome:
            attended, metric = block.attn(block.norm1(reference), size=None)
            reference = reference + attended
            remove = min(reference.shape[1] // 2, max(0, reference.shape[1] - m.frame_tokens))
            if remove:
                merge, _ = bipartite_soft_matching(metric, remove)
                reference, _ = merge_wavg(merge, reference, None)
            reference = block.forward_mlp(reference)
            if reference.shape[1] <= m.frame_tokens:
                break
        torch.testing.assert_close(m._compress(raw), reference, atol=1e-6, rtol=1e-5)

    def test_bfloat16_merge_single_rounding_and_gradient(self):
        from src.memory.tome.merge import bipartite_soft_matching, merge_wavg
        torch.manual_seed(8)
        metric = torch.randn(2, 12, 4)
        features = torch.randn(2, 12, 16, dtype=torch.bfloat16, requires_grad=True)
        stable, _ = bipartite_soft_matching(metric, 3, deterministic=True)
        upstream, _ = bipartite_soft_matching(metric, 3)
        actual, _ = merge_wavg(stable, features)
        reference, _ = merge_wavg(upstream, features.float())
        torch.testing.assert_close(actual, reference.to(torch.bfloat16), atol=0, rtol=0)
        repeated, _ = merge_wavg(stable, features)
        torch.testing.assert_close(actual, repeated, atol=0, rtol=0)
        actual.float().square().sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(float(features.grad.abs().sum()), 0)

    def test_isolation_order_and_duplicate(self):
        m = self.make(consolidation='fifo')
        f = torch.randn(8, 16)
        m.write('a', 5, (0, 0, 0), f)
        m.write('a', 5, (0, 0, 0), f + 10)
        torch.testing.assert_close(m.banks['a'][5][1], f)
        with self.assertRaises(ValueError):
            m.write('a', 4, (0, 0, 0), f)
        with self.assertRaises(ValueError):
            m.read('a', 4, (0, 0, 0))
        with self.assertRaises(ValueError):
            m.read('other', 5, (0, 0, 0))


if __name__ == '__main__':
    unittest.main()

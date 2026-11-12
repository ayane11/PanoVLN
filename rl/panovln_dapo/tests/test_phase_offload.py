from types import SimpleNamespace
from unittest.mock import MagicMock,patch
import unittest
import torch
from panovln_dapo.phase_offload import move_actor_between_phases


class PhaseOffloadTests(unittest.TestCase):
    def model(self):
        flat=torch.nn.Parameter(torch.ones(4));flat._local_shard=flat.detach()
        handle=SimpleNamespace(flat_param=flat,_offload_params=False)
        def move(device,non_blocking):flat.data=flat.detach().clone().to(device)
        handle.flat_param_to=move
        model=SimpleNamespace(_is_root=True,_all_handles=[handle],
                              parameters=lambda:[flat],module=SimpleNamespace(panovggt=MagicMock()))
        return model,flat,handle
    def test_local_shard_tracks_new_storage_and_geometry_moves(self):
        model,flat,_=self.model();old=flat.data_ptr()
        with patch('panovln_dapo.phase_offload._lazy_init'),patch('panovln_dapo.phase_offload.torch.cuda.empty_cache'):
            move_actor_between_phases(model,'cpu')
        self.assertNotEqual(flat.data_ptr(),old)
        self.assertEqual(flat.data_ptr(),flat._local_shard.data_ptr())
        model.module.panovggt.to.assert_called_once_with('cpu')
    def test_gradients_unsharded_storage_and_mixed_offload_rejected(self):
        with patch('panovln_dapo.phase_offload._lazy_init'):
            model,flat,handle=self.model();flat.grad=torch.ones_like(flat)
            with self.assertRaisesRegex(ValueError,'gradients'):move_actor_between_phases(model,'cpu')
            flat.grad=None;flat._local_shard=flat.detach().clone()
            with self.assertRaisesRegex(ValueError,'sharded'):move_actor_between_phases(model,'cpu')
            handle._offload_params=True
            with self.assertRaisesRegex(ValueError,'Per-forward'):move_actor_between_phases(model,'cpu')

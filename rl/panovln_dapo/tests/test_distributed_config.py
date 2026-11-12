"""Construction contract checks; real hybrid FSDP still needs multi-GPU validation."""
from contextlib import ExitStack
import unittest
from unittest.mock import MagicMock,patch
import torch
from torch.distributed.fsdp import ShardingStrategy
from panovln_dapo.distributed import load_actor


class DistributedConfigTests(unittest.TestCase):
    def construct(self,size,checkpointing=True,prefetch=False):
        stack=ExitStack();self.addCleanup(stack.close)
        stack.enter_context(patch('panovln_dapo.distributed.dist.get_world_size',return_value=4))
        factory=stack.enter_context(patch('panovln_dapo.distributed.Qwen3_5ForConditionalGenerationForPanoVLN.from_pretrained'))
        model=factory.return_value
        model.modules.return_value=[type('Qwen3_5DecoderLayer',(),{})()]
        wrapper=stack.enter_context(patch('panovln_dapo.distributed.FSDP'))
        mesh=stack.enter_context(patch('panovln_dapo.distributed.init_device_mesh'))
        load_actor('checkpoint',torch.device('cuda',0),fsdp_size=size,
                   gradient_checkpointing=checkpointing,forward_prefetch=prefetch)
        return model,wrapper,mesh

    def test_full_shard_defaults(self):
        model,wrapper,mesh=self.construct(-1)
        mesh.assert_not_called()
        self.assertEqual(wrapper.call_args.kwargs['sharding_strategy'],ShardingStrategy.FULL_SHARD)
        model.gradient_checkpointing_enable.assert_called_once()
        self.assertFalse(wrapper.call_args.kwargs['forward_prefetch'])

    def test_hybrid_mesh_and_checkpoint_switch(self):
        model,wrapper,mesh=self.construct(2,False,True)
        mesh.assert_called_once_with('cuda',(2,2),mesh_dim_names=('replicate','shard'))
        self.assertEqual(wrapper.call_args.kwargs['sharding_strategy'],ShardingStrategy.HYBRID_SHARD)
        model.gradient_checkpointing_disable.assert_called_once()
        model.gradient_checkpointing_enable.assert_not_called()
        self.assertTrue(wrapper.call_args.kwargs['forward_prefetch'])

    def test_invalid_sizes_fail_before_model_loading(self):
        with patch('panovln_dapo.distributed.dist.get_world_size',return_value=4), patch('panovln_dapo.distributed.Qwen3_5ForConditionalGenerationForPanoVLN.from_pretrained') as load:
            for size in (0,-2,3,8):
                with self.assertRaises(ValueError):load_actor('checkpoint','cuda:0',fsdp_size=size)
            load.assert_not_called()

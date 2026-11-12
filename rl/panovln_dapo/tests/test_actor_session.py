"""Run the real update loop on CPU to check persistent Adam and policy binding."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import torch
from panovln_dapo.actor import ActorSession


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.logits=torch.nn.Parameter(torch.tensor([.2,-.1]))
    def clip_grad_norm_(self,max_norm):
        return torch.nn.utils.clip_grad_norm_(self.parameters(),max_norm)


class ActorSessionTests(unittest.TestCase):
    def test_two_updates_keep_adam_and_reject_stale_batch(self):
        session=ActorSession.__new__(ActorSession)
        session.model=TinyPolicy();session.processor=None
        session.optimizer=torch.optim.AdamW(session.model.parameters(),lr=.01)
        session.rank=0;session.world=1;session.device=torch.device('cpu')
        session.policy_version=0;session.completed_optimizer_steps=0
        args=SimpleNamespace(log_prob_micro_batch_size=2,ppo_micro_batch_size=2,precision='fp32',
                             ppo_epochs=1,mini_batch_decisions=2,entropy_coeff=0.,clip_low=.2,
                             clip_high=.28,dual_clip=10.,optimizer_offload=False)
        def replay(model,processor,rows,device,**kwargs):
            return [model.logits.log_softmax(0)[row['target']].reshape(1) for row in rows]
        with patch('panovln_dapo.actor.replay_microbatch',side_effect=replay), patch('panovln_dapo.actor.dist.all_reduce'):
            for version in range(2):
                rows=[]
                for target,advantage in ((0,1.),(1,-1.)):
                    value=session.model.logits.detach().log_softmax(0)[target].item()
                    rows.append(dict(target=target,advantage=advantage,generation=SimpleNamespace(
                        response_logprobs=[value],response_ids=[target],response_mask=[1])))
                metrics=session.update({'policy_version':version},rows,args)
                self.assertEqual(metrics['policy_version'],version+1)
                self.assertEqual(session.completed_optimizer_steps,version+1)
                state=session.optimizer.state[session.model.logits]
                self.assertEqual(state['step'].item(),version+1)
            self.assertGreater(session.optimizer.state[session.model.logits]['exp_avg'].abs().sum().item(),0)
            with self.assertRaisesRegex(ValueError,'stale'):
                session.update({'policy_version':0},rows,args)

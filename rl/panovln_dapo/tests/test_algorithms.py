"""Compare ported objectives directly against current SimpleVLN source on CPU."""
import ast
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from panovln_dapo.algorithms import trajectory_grpo_advantage, policy_loss_sum

SOURCE = Path(__file__).resolve().parents[4] / 'SimpleVLN' / 'RL'


def source_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class ObjectiveTests(unittest.TestCase):
    def test_trajectory_advantage_matches_source(self):
        ref = source_function(SOURCE/'recipe/vln_simplevln/patches.py', '_traj_grpo_advantage',
                              {'torch':torch,'defaultdict':defaultdict})
        rewards = torch.tensor([[0.,1.],[0.,0.],[0.,0.],[0.,0.]])
        mask = torch.tensor([[1.,1.],[1.,1.],[1.,0.],[0.,0.]])
        groups = ['g','g','g','pad']; trajectories=['win','lose','lose','pad']
        weights = [1.,.5,.5,0.]
        actual = trajectory_grpo_advantage(rewards,mask,groups,trajectories,weights)[0]
        expected = ref(rewards,mask,groups,trajectories,weights)[0]
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertGreater(actual[0,0],0)
        self.assertLess(actual[1,0],0)
        self.assertEqual(actual[-1].abs().sum().item(),0)

    def test_loss_and_gradients_match_source(self):
        masked_mean=lambda values,mask: (values*mask).sum()/mask.sum()
        namespace={'torch':torch, 'verl_F':SimpleNamespace(masked_mean=masked_mean),
                   'agg_loss':lambda loss_mat,loss_mask,loss_agg_mode:masked_mean(loss_mat,loss_mask)}
        ref=source_function(SOURCE/'verl/trainer/ppo/core_algos.py','compute_policy_loss',namespace)
        rng=torch.Generator().manual_seed(42)
        for _ in range(20):
            old=torch.randn(4,7,generator=rng)
            new=(old+torch.randn(4,7,generator=rng)*3).requires_grad_()
            other=new.detach().clone().requires_grad_()
            adv=torch.randn(4,7,generator=rng)
            mask=torch.randint(0,2,(4,7),generator=rng).float(); mask[0,0]=1
            numerator,count=policy_loss_sum(old,new,adv,mask)
            actual=numerator/count
            expected=ref(old,other,adv,mask,cliprange=.2,cliprange_low=.2,cliprange_high=.28,clip_ratio_c=10.)[0]
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            actual.backward();expected.backward()
            torch.testing.assert_close(new.grad,other.grad,rtol=0,atol=0)

    def test_microbatch_normalization(self):
        old=torch.zeros(3,2);new=torch.zeros(3,2,requires_grad=True)
        adv=torch.tensor([[1.,1.],[-1.,-1.],[2.,2.]])
        mask=torch.tensor([[1.,0.],[1.,1.],[0.,0.]])
        full,n=policy_loss_sum(old,new,adv,mask)
        sums=[policy_loss_sum(old[i:i+1],new[i:i+1],adv[i:i+1],mask[i:i+1])[0] for i in range(3)]
        torch.testing.assert_close(full/n,sum(sums)/n)

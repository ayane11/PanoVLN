import unittest
import torch
from panovln_dapo.optimizer import step_adamw_with_parameter_offload


class ParameterOffloadTests(unittest.TestCase):
    def test_matches_adamw_parameters_moments_and_steps(self):
        left=[torch.nn.Parameter(torch.tensor([1.,2.])),torch.nn.Parameter(torch.tensor([3.]))]
        right=[torch.nn.Parameter(p.detach().clone()) for p in left]
        def make(params):
            return torch.optim.AdamW([{'params':[params[0]],'lr':.01},{'params':[params[1]],'lr':.03,'weight_decay':.02}],
                                     foreach=False,amsgrad=True)
        reference,offloaded=make(left),make(right)
        original_groups=offloaded.param_groups
        for step in range(4):
            for index,(a,b) in enumerate(zip(left,right)):
                grad=None if step==2 and index==1 else torch.full_like(a,.2*(step+1)*(index+1))
                a.grad=grad;b.grad=None if grad is None else grad.clone()
            reference.step();step_adamw_with_parameter_offload(offloaded)
            self.assertIs(offloaded.param_groups,original_groups)
            for a,b in zip(left,right):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
                for key,value in reference.state[a].items():
                    torch.testing.assert_close(value,offloaded.state[b][key],atol=0,rtol=0)
    def test_hooks_rejected_before_update(self):
        p=torch.nn.Parameter(torch.ones(1));p.grad=torch.ones(1)
        optimizer=torch.optim.AdamW([p],foreach=False)
        optimizer.register_step_post_hook(lambda *args:None)
        with self.assertRaisesRegex(ValueError,'hooks'):step_adamw_with_parameter_offload(optimizer)
        self.assertTrue(torch.equal(p,torch.ones(1)))

    def test_cpu_restore_preserves_all_state_and_next_update(self):
        from panovln_dapo.optimizer import load_adamw_state_on_cpu
        a=torch.nn.Parameter(torch.tensor([1.,2.]));b=torch.nn.Parameter(a.detach().clone())
        first=torch.optim.AdamW([a],lr=.01,foreach=False,amsgrad=True)
        second=torch.optim.AdamW([b],lr=.01,foreach=False,amsgrad=True)
        a.grad=torch.tensor([.2,-.3]);first.step();b.data.copy_(a)
        import copy
        saved=copy.deepcopy(first.state_dict())
        load_adamw_state_on_cpu(second,saved)
        a.grad=torch.tensor([.4,.1]);b.grad=a.grad.clone()
        first.step();step_adamw_with_parameter_offload(second)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        for key,value in first.state[a].items():torch.testing.assert_close(value,second.state[b][key],atol=0,rtol=0)
        bad=copy.deepcopy(saved);bad['state'][0]['exp_avg']=torch.ones(3)
        with self.assertRaisesRegex(ValueError,'shape'):load_adamw_state_on_cpu(second,bad)

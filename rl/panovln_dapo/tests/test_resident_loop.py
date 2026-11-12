import unittest
from panovln_dapo.resident_loop import ResidentLoop


class ResidentLoopTests(unittest.TestCase):
    def callbacks(self,limit,fail_at=None):
        calls=[];exports=[]
        def collect(version):
            return {'policy_version':version,'cursor':version+1} if version<limit else None
        def update(payload):
            calls.append(payload['policy_version'])
            if payload['policy_version']==fail_at:raise RuntimeError('interrupted')
            return dict(policy_version=payload['policy_version']+1,optimizer_steps=2)
        def export(payload,metrics):exports.append((payload.copy(),metrics.copy()))
        return collect,update,export,calls,exports

    def test_five_updates_and_final_tail(self):
        c,u,e,calls,exports=self.callbacks(12)
        result=ResidentLoop(policy_version=0).run(c,u,e)
        self.assertEqual(calls,list(range(12)))
        self.assertEqual([m['policy_version'] for _,m in exports],[5,10,12])
        self.assertEqual([m['optimizer_steps'] for _,m in exports],[10,10,4])
        self.assertEqual([p['cursor'] for p,_ in exports],[5,10,12])
        self.assertEqual(result.durable_optimizer_steps,24)

    def test_interrupt_and_replay_from_last_durable(self):
        c,u,e,calls,exports=self.callbacks(12,9)
        loop=ResidentLoop(policy_version=0)
        with self.assertRaises(RuntimeError):loop.run(c,u,e)
        self.assertEqual(loop.progress.durable_version,5)
        self.assertEqual([m['policy_version'] for _,m in exports],[5])
        with self.assertRaisesRegex(RuntimeError,'cannot be reused'):loop.run(c,u,e)
        c,u,e,calls,exports=self.callbacks(12)
        ResidentLoop(policy_version=5,optimizer_steps=10).run(c,u,e)
        self.assertEqual(calls,list(range(5,12)))
        self.assertEqual(exports[-1][1]['cumulative_optimizer_steps'],24)

    def test_export_failure_never_advances_durable_version(self):
        c,u,_,_,_=self.callbacks(5)
        def fail(*args):raise OSError('disk full')
        loop=ResidentLoop(policy_version=0)
        with self.assertRaises(OSError):loop.run(c,u,fail)
        self.assertEqual(loop.progress.durable_version,0)

    def test_update_budget_flushes_tail(self):
        c,u,e,calls,exports=self.callbacks(20)
        ResidentLoop(policy_version=0).run(c,u,e,max_updates=3)
        self.assertEqual(len(calls),3)
        self.assertEqual(exports[-1][1]['policy_version'],3)

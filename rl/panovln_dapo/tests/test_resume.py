import copy
import unittest
from panovln_dapo.resume import validate_continuation


class ResumeContractTests(unittest.TestCase):
    def setUp(self):
        cursor=dict(fingerprint='dataset',seed=42,epochs=20,shuffle=True,epoch=0,offset=16)
        self.saved=dict(cursor=cursor,policy_version=1,rng_by_rank=[{},{}])
        self.payload=dict(cursor_before=copy.deepcopy(cursor),cursor={**cursor,'offset':32},policy_version=1)

    def test_exact_next_collection_and_epoch_boundary(self):
        validate_continuation(self.saved,self.payload,2)
        self.payload['cursor'].update(epoch=1,offset=0)
        validate_continuation(self.saved,self.payload,2)

    def test_skipped_or_repeated_data_rejected(self):
        self.payload['cursor_before']['offset']=17
        with self.assertRaisesRegex(ValueError,'begin'):validate_continuation(self.saved,self.payload,2)
        self.payload['cursor_before']['offset']=16
        self.payload['cursor']['offset']=16
        with self.assertRaisesRegex(ValueError,'advance'):validate_continuation(self.saved,self.payload,2)

    def test_policy_or_world_change_rejected(self):
        self.payload['policy_version']=0
        with self.assertRaisesRegex(ValueError,'policy'):validate_continuation(self.saved,self.payload,2)
        self.payload['policy_version']=1
        with self.assertRaisesRegex(ValueError,'world'):validate_continuation(self.saved,self.payload,4)

    def test_explicit_reshard_preserves_data_contract_and_reseeds(self):
        from unittest.mock import patch
        from panovln_dapo.resume import restore_rng
        validate_continuation(self.saved,self.payload,4,allow_world_size_change=True)
        self.payload['cursor_before']['offset']=15
        with self.assertRaisesRegex(ValueError,'begin'):
            validate_continuation(self.saved,self.payload,4,allow_world_size_change=True)
        with patch('torch.manual_seed'),patch('torch.cuda.manual_seed'),patch('random.seed'):
            a=restore_rng(self.saved,4,0,allow_world_size_change=True)
            b=restore_rng(self.saved,4,0,allow_world_size_change=True)
            c=restore_rng(self.saved,4,1,allow_world_size_change=True)
        self.assertEqual(a,b);self.assertNotEqual(a['rank_seed'],c['rank_seed'])
        self.assertEqual(a['mode'],'deterministic_reseed_after_reshard')

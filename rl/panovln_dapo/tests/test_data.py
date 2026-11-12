import copy
import json
from pathlib import Path
import tempfile
import unittest
from panovln_dapo.data import annotation_rows, restore_annotation


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "annotation.json"
        self.episode = dict(episode_id=1, scene_id="mp3d/a/a.glb", start_position=[0, 0, 0],
                            start_rotation=[0, 0, 0, 1], goals=[{"position": [1, 0, 0]}],
                            reference_path=[[0, 0, 0], [1, 0, 0]],
                            instruction={"instruction_text": "Go left"})
        self.payload = {"instruction_vocab": {"word_list": ["Go", "left"]}, "episodes": [self.episode]}

    def rows(self):
        self.path.write_text(json.dumps(self.payload))
        return annotation_rows(self.path, dataset_name="r2r", split="train", dataset_type="R2RVLNCE-v1")

    def test_lossless_roundtrip(self):
        self.assertEqual(restore_annotation(self.rows()), self.payload)

    def test_duplicate_rejected(self):
        self.payload["episodes"].append(copy.deepcopy(self.episode))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.rows()

    def test_scene_full_path_is_identity(self):
        e = copy.deepcopy(self.episode)
        e["scene_id"] = "other/a/a.glb"
        self.payload["episodes"].append(e)
        self.assertEqual(len(self.rows()), 2)

    def test_metadata_required(self):
        self.payload.pop("instruction_vocab")
        with self.assertRaisesRegex(ValueError, "instruction_vocab"):
            self.rows()

    def test_corrupt_routing_rejected(self):
        rows = self.rows()
        rows[0]["extra_info"]["instruction"] = "wrong"
        with self.assertRaisesRegex(ValueError, "routing"):
            restore_annotation(rows)

    def test_corrupt_episode_rejected(self):
        rows = self.rows()
        e = json.loads(rows[0]["episode"])
        e["goals"][0]["position"][0] = 2
        rows[0]["episode"] = json.dumps(e)
        with self.assertRaisesRegex(ValueError, "integrity"):
            restore_annotation(rows)

    def test_nonfinite_goal_rejected(self):
        self.episode["goals"][0]["position"][0] = float("nan")
        with self.assertRaises(ValueError):
            self.rows()

    def test_mixed_sources_rejected(self):
        rows = self.rows()
        other = copy.deepcopy(rows[0]); other["source_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "Partition"):
            restore_annotation(rows + [other])


class RxRTimingTests(unittest.TestCase):
    setUp = DataContractTests.setUp

    def convert(self):
        self.path.write_text(json.dumps(self.payload))
        return annotation_rows(self.path,dataset_name='rxr',split='train',dataset_type='RxRVLNCE-v1')

    def test_optional_nan_timestamps_preserved(self):
        import math
        self.episode['instruction']['timed_instruction']=[{'start_time':float('nan'),'end_time':float('nan'),'word':'Go'}]
        result=restore_annotation(self.convert())
        self.assertTrue(math.isnan(result['episodes'][0]['instruction']['timed_instruction'][0]['start_time']))
        self.assertEqual(result['episodes'][0]['instruction']['instruction_text'],'Go left')
        self.path.write_text(json.dumps(self.payload))
        with self.assertRaises(ValueError):
            annotation_rows(self.path,dataset_name='r2r',split='train',dataset_type='R2RVLNCE-v1')

    def test_other_nonfinite_fields_still_rejected(self):
        self.episode['instruction']['edit_distance']=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'): self.convert()
        del self.episode['instruction']['edit_distance']
        self.episode['instruction']['timed_instruction']=[{'start_time':float('inf')}]
        with self.assertRaisesRegex(ValueError,'Nonfinite'): self.convert()

    def test_explicit_language_subset(self):
        self.episode['instruction']['language']='en-US'
        other=copy.deepcopy(self.episode)
        other['episode_id']=2;other['instruction']['language']='hi-IN'
        self.payload['episodes'].append(other)
        self.path.write_text(json.dumps(self.payload))
        rows=annotation_rows(self.path,dataset_name='rxr',split='train',dataset_type='RxRVLNCE-v1',languages=['en-US','en-IN'])
        self.assertEqual(len(rows),1)
        self.assertEqual(restore_annotation(rows)['episodes'][0]['instruction']['language'],'en-US')
        with self.assertRaisesRegex(ValueError,'No episodes'):
            annotation_rows(self.path,dataset_name='rxr',split='train',dataset_type='RxRVLNCE-v1',languages=['missing'])


if __name__ == "__main__":
    unittest.main()

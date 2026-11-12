import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from panovln_dapo.collect import write_collection_json


class CollectionDiskTests(unittest.TestCase):
    def test_payload_size_is_reserved_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trajectory.json'; payload={'instruction':'向左走'}
            size=len((json.dumps(payload,ensure_ascii=False)+'\n').encode())
            with patch('panovln_dapo.collect.shutil.disk_usage',return_value=SimpleNamespace(free=70*1024**3+size-1)):
                with self.assertRaisesRegex(RuntimeError,'reserve'):write_collection_json(path,payload,min_free_gib=70)
            self.assertFalse(path.exists())
            with patch('panovln_dapo.collect.shutil.disk_usage',return_value=SimpleNamespace(free=70*1024**3+size)):
                write_collection_json(path,payload,min_free_gib=70)
            self.assertEqual(json.loads(path.read_text()),payload)

    def test_existing_trajectory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'batch.json';path.write_text('existing')
            with self.assertRaises(FileExistsError):write_collection_json(path,{'new':True})
            self.assertEqual(path.read_text(),'existing')

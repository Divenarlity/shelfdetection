import tempfile, unittest
from pathlib import Path
from src.reporter import unique_output_path
from src.store_map import load_store_map


class StoreMapTests(unittest.TestCase):
    def test_missing_camera(self):
        sm=load_store_map(Path("configs/store_map.example.yaml"))
        with self.assertRaises(KeyError): sm.camera("MISSING")

    def test_broken_yaml(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"bad.yaml"; p.write_text("market_id: [",encoding="utf-8")
            with self.assertRaises(ValueError): load_store_map(p)

    def test_missing_fields(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"bad.yaml"; p.write_text("market_id: X\ncameras:\n  CAM: {}\n",encoding="utf-8")
            with self.assertRaises(ValueError): load_store_map(p)

    def test_unique_output_policy(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d); (folder/"a.jpg").touch()
            self.assertEqual(unique_output_path(folder, "a", ".jpg").name, "a_1.jpg")

"""Meaningful archive-integrity, hierarchy and safe-dispatch checks."""
import ast
import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from leafshift_cp.verification import verify
from leafshift_cp.workflows import WORKFLOWS, plan, run


class ResearchRepositoryTests(unittest.TestCase):
    def test_archived_evidence_and_source_hashes(self):
        report=verify()
        self.assertEqual([c for c in report['checks'] if c['status']=='fail'],[])

    def test_all_python_sources_parse(self):
        for path in ROOT.rglob('*.py'):
            if not {'.venv', '.git', '.cache', '__pycache__', 'runs'}.intersection(path.relative_to(ROOT).parts):
                with self.subTest(source=str(path.relative_to(ROOT))):
                    ast.parse(path.read_bytes(),filename=str(path))

    def test_scientific_runtime_imports(self):
        for module in ['numpy','pandas','scipy','sklearn','matplotlib','PIL','openpyxl']:
            with self.subTest(module=module):self.assertIsNotNone(importlib.import_module(module))

    def test_routes_reference_real_original_sources(self):
        for name,spec in WORKFLOWS.items():
            for path in spec['scripts']:self.assertTrue((ROOT/path).is_file(),path)
            self.assertEqual(plan(name)['workflow'],name)

    def test_missing_manifest_is_not_fabricated(self):
        self.assertIn('data/processed/forecasting/forecast_windows.csv',plan('conformal')['missing_inputs'])

    def test_expensive_route_requires_explicit_execution_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'research-run'
            with self.assertRaisesRegex(ValueError,'allow-expensive'):
                run('adaptive-cqr',output)
            self.assertFalse(output.exists())

    def test_existing_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'new directory'):
                run('tables',Path(directory))

    def test_nested_output_cannot_recursively_copy_repository(self):
        with self.assertRaisesRegex(ValueError,'recursive copying'):
            run('tables',ROOT/'unapproved-recursive-output')
        self.assertFalse((ROOT/'unapproved-recursive-output').exists())

    def test_original_feature_constructor_and_weighting(self):
        # Unit-test fixture, not a research observation or manufactured evidence.
        import numpy as np
        path=ROOT/'src/E3/run_cqr_baselines.py'
        spec=importlib.util.spec_from_file_location('original_cqr',path)
        original=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        times=np.array([-30,-24,-18,-12,-6,0],dtype=float)
        areas=np.array([100,110,120,130,140,150],dtype=float)
        features=original.build_features(times,areas,24)
        self.assertEqual(features.shape,(29,))
        np.testing.assert_array_equal(features[:6],areas)
        self.assertEqual(features[5],150)
        with self.assertRaises(ValueError):original.build_features(np.zeros(6),areas,24)


if __name__=='__main__':unittest.main()

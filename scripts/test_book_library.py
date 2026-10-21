"""Module-generation regressions; runtime checks use the existing PyTorch environment.

Run: python3 -m unittest discover -s scripts -p 'test_book_library.py' -v
"""
import ast
import importlib
import importlib.util
from pathlib import Path
import pickle
import random
import sys
import tempfile
import types
import unittest

from book_library import (ROOT, build_modules, chapter_paths, defined_names, make_library,
                          package_files, saved_blocks)


class GenerationTests(unittest.TestCase):
    def test_modules_have_explicit_backward_dependencies(self):
        modules = build_modules(chapter_paths())
        earlier = set()
        for module in modules:
            with self.subTest(module=module.name):
                self.assertLessEqual(set(module.dependencies), earlier)
                tree = ast.parse(module.source)
                self.assertFalse(any(isinstance(n, ast.Name) and n.id == 'arl' for n in ast.walk(tree)))
                self.assertEqual(set(defined_names(tree)) - {'__all__'}, set(module.exports))
            earlier.add(module.name)

    def test_generated_files_are_current_and_repeatable(self):
        modules = build_modules(chapter_paths())
        files = package_files(modules)
        self.assertEqual(files, package_files(build_modules(chapter_paths())))
        for name, source in files.items():
            with self.subTest(file=name):
                self.assertEqual((ROOT / 'agentrl' / name).read_text(), source)
        init = ast.parse(files['__init__.py'])
        self.assertFalse(any(isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in ast.walk(init)))

    def test_future_and_dynamic_api_references_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / '02-example.md'
            for expression in ('arl.not_yet_defined()', 'getattr(arl, "x")()'):
                with self.subTest(expression=expression):
                    path.write_text('```python\n#@save\nimport agentrl as arl\n'
                                    f'def example():\n    return {expression}\n```\n')
                    with self.assertRaises(ValueError):
                        build_modules([path])

    def test_comprehension_variables_are_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / '01-example.md'
            path.write_text('```python\n#@save\nVALUES = [i for i in range(3)]\n'
                            'def example():\n    return [str(item) for item in VALUES]\n```\n')
            spec, = build_modules([path])
            namespace = {}
            exec(spec.source, namespace)
            self.assertEqual(namespace['example'](), ['0', '1', '2'])

    def test_missing_global_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / '01-example.md'
            path.write_text('```python\n#@save\ndef example():\n    return missing_name\n```\n')
            with self.assertRaisesRegex(ValueError, 'unresolved saved global missing_name'):
                build_modules([path])


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is available on the check host')
class RuntimeTests(unittest.TestCase):
    def tearDown(self):
        for name in list(sys.modules):
            if name == 'agentrl' or name.startswith('agentrl.'):
                del sys.modules[name]

    def test_public_api_and_chapter_visibility(self):
        paths = chapter_paths()
        full = make_library(paths)
        for spec in build_modules(paths):
            module = importlib.import_module('agentrl.' + spec.name)
            for name in spec.exports:
                self.assertIs(getattr(full, name), getattr(module, name))
        partial = make_library([p for p in paths if p.name[:2] < '05'])
        self.assertTrue(hasattr(partial, 'BugFixEnv'))
        self.assertFalse(hasattr(partial, 'Trajectory'))
        self.assertFalse(hasattr(partial, 'RLConfig'))
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module('agentrl.grpo')
        self.assertNotIn('agentrl.grpo', sys.modules)

    def test_persistent_import_and_pickle(self):
        arl = importlib.import_module('agentrl')
        self.assertIs(arl.TinyAgent, arl.policy.TinyAgent)
        task = arl.make_tasks('train')[0]
        self.assertEqual(pickle.loads(pickle.dumps(task)), task)
        self.assertEqual(type(task).__module__, 'agentrl.environments')

    def test_mutable_configuration_has_a_single_owner(self):
        arl = make_library(chapter_paths())
        task = arl.make_tasks('train')[0]
        old = arl.grpo.FN_NAMES
        try:
            arl.grpo.FN_NAMES = ['new_function_name']
            self.assertEqual(arl.randomize_names(task, random.Random(0)).fn_name, 'new_function_name')
        finally:
            arl.grpo.FN_NAMES = old

    def test_training_matches_legacy_shared_namespace(self):
        import torch
        torch.set_num_threads(4)
        # The previous implementation is a reference oracle for this refactor.
        legacy = types.ModuleType('agentrl')
        sys.modules['agentrl'] = legacy
        for path in chapter_paths():
            for code in saved_blocks(path):
                exec(compile(code, str(path), 'exec'), legacy.__dict__)
        legacy_task = pickle.dumps(legacy.make_tasks('train')[0])
        modular = make_library(chapter_paths())
        self.assertEqual(pickle.loads(legacy_task), modular.make_tasks('train')[0])

        def exercise(arl):
            tasks = arl.make_tasks('train')
            tok = arl.build_tokenizer(arl.make_tasks('all'))
            base = arl.make_rl_base(tok, seed=0)
            cfg = arl.RLConfig(prompts_per_step=2, group_size=4)
            history = arl.train_rl(base, tasks, tok, cfg, 2)
            trajs = arl.sample_groups(base, lambda: arl.make_env(cfg), tasks[:2], tok, cfg,
                                      torch.Generator().manual_seed(7))
            batch = arl.collate(trajs, tok.pad_id)
            rewards = arl.outcome_rewards(batch['rewards'], batch['action_mask'])
            returns = arl.returns_to_go(rewards, batch['action_mask'])
            packed = arl.pack_sequences(trajs, base.max_len, tok.pad_id)
            logits = arl.packed_logits(base, packed['ids'], packed['position_ids'], packed['segment_ids'])
            return dict(history=history, weights=base.state_dict(), batch=batch, returns=returns,
                        packed_logits=logits.detach(),
                        counts=arl.sample_counts(base, tasks[:2], tok, 2, cfg, torch.Generator().manual_seed(8)),
                        duration=arl.agent_episode_durations(16, torch.Generator().manual_seed(3)))

        def equal(left, right):
            if isinstance(left, torch.Tensor):
                self.assertTrue(torch.equal(left, right))
            elif isinstance(left, dict):
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    equal(left[key], right[key])
            elif isinstance(left, (list, tuple)):
                self.assertEqual(len(left), len(right))
                for a, b in zip(left, right):
                    equal(a, b)
            else:
                self.assertEqual(left, right)

        equal(exercise(legacy), exercise(modular))


if __name__ == '__main__':
    unittest.main()

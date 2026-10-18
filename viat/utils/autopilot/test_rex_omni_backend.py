"""Offline tests for the optional Rex-Omni Autopilot adapter."""

import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from . import rex_omni_backend


class _FakeWrapper:
    instances = []
    response = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.inference_kwargs = None
        type(self).instances.append(self)

    def inference(self, **kwargs):
        self.inference_kwargs = kwargs
        return type(self).response


class _FailingWrapper:
    def __init__(self, **kwargs):
        raise RuntimeError("fake model construction failed")


class RexOmniBackendTests(unittest.TestCase):
    def setUp(self):
        self.checkpoint = tempfile.TemporaryDirectory()
        self.addCleanup(self.checkpoint.cleanup)
        self.resolve_patch = mock.patch.object(
            rex_omni_backend, "resolve_hf", return_value=self.checkpoint.name
        )
        self.resolve_patch.start()
        self.addCleanup(self.resolve_patch.stop)
        _FakeWrapper.instances = []
        _FakeWrapper.response = []

    def _install_rex_module(self, wrapper):
        fake_module = types.ModuleType("rex_omni")
        fake_module.RexOmniWrapper = wrapper
        missing = object()
        previous = sys.modules.get("rex_omni", missing)
        sys.modules["rex_omni"] = fake_module

        def restore():
            if previous is missing:
                sys.modules.pop("rex_omni", None)
            else:
                sys.modules["rex_omni"] = previous

        self.addCleanup(restore)

    def test_list_response_parsing_and_bgr_to_rgb(self):
        self._install_rex_module(_FakeWrapper)
        _FakeWrapper.response = [
            {
                "success": True,
                "extracted_predictions": {
                    "traffic car": [
                        {"type": "box", "coords": [-2, -1, 3, 4]},
                        {"type": "point", "coords": [1, 1]},
                        {"type": "box", "coords": [1, 1, 99, 99]},
                        {"type": "box", "coords": [float("nan"), 0, 2, 2]},
                        {"type": "box", "coords": [3, 2, 2, 3]},
                        {"type": "box", "coords": [0, 0, 1]},
                        "not a record",
                    ],
                    "person": [{"type": "box", "coords": [0, 0, 2, 2]}],
                },
            }
        ]

        backend = rex_omni_backend.RexOmniBackend("IDEA-Research/Rex-Omni")
        self.assertEqual((True, "ok"), backend.load())
        backend.set_prompts(["traffic car", "person"])

        # OpenCV's BGR pixel [10, 20, 30] must reach Rex as RGB [30, 20, 10].
        image = np.array(
            [
                [[10, 20, 30], [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
                [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23], [24, 25, 26]],
                [[27, 28, 29], [30, 31, 32], [33, 34, 35], [36, 37, 38], [39, 40, 41]],
                [[42, 43, 44], [45, 46, 47], [48, 49, 50], [51, 52, 53], [54, 55, 56]],
            ],
            dtype=np.uint8,
        )
        detections = backend.predict(image)

        self.assertTrue(backend.constant)
        self.assertEqual(1.0, backend.ref_score)
        self.assertEqual(3, len(detections))
        self.assertEqual(
            [
                ([0.0, 0.0, 3.0, 4.0], 1.0, "traffic car"),
                ([1.0, 1.0, 5.0, 4.0], 1.0, "traffic car"),
                ([0.0, 0.0, 2.0, 2.0], 1.0, "person"),
            ],
            detections,
        )

        call = _FakeWrapper.instances[0].inference_kwargs
        self.assertEqual("detection", call["task"])
        self.assertEqual(["traffic car", "person"], call["categories"])
        self.assertEqual((5, 4), call["images"].size)
        self.assertEqual("RGB", call["images"].mode)
        self.assertEqual([30, 20, 10], list(call["images"].getpixel((0, 0))))

    def test_direct_dict_response_and_default_box_type(self):
        self._install_rex_module(_FakeWrapper)
        _FakeWrapper.response = {
            "extracted_predictions": {
                "cup": [{"coords": [1, 2, 3, 4]}],
            }
        }
        backend = rex_omni_backend.RexOmniBackend("IDEA-Research/Rex-Omni")
        backend.load()
        backend.set_prompts(["cup"])

        detections = backend.predict(np.zeros((8, 9, 3), dtype=np.uint8))
        self.assertEqual([([1.0, 2.0, 3.0, 4.0], 1.0, "cup")], detections)

    def test_awq_prefers_vllm_when_available(self):
        self._install_rex_module(_FakeWrapper)
        with mock.patch.object(rex_omni_backend, "_vllm_available", return_value=True):
            backend = rex_omni_backend.RexOmniBackend("IDEA-Research/Rex-Omni-AWQ")
            self.assertEqual((True, "ok"), backend.load())

        options = _FakeWrapper.instances[0].kwargs
        self.assertEqual(self.checkpoint.name, options["model_path"])
        self.assertEqual("vllm", options["backend"])
        self.assertEqual("awq", options["quantization"])
        self.assertNotIn("local_files_only", options)

    def test_awq_fails_clearly_without_vllm(self):
        self._install_rex_module(_FakeWrapper)
        with mock.patch.object(rex_omni_backend, "_vllm_available", return_value=False):
            backend = rex_omni_backend.RexOmniBackend("IDEA-Research/Rex-Omni-AWQ")
            ok, message = backend.load()

        self.assertFalse(ok)
        self.assertIn("requires vLLM", message)
        self.assertEqual([], _FakeWrapper.instances)

    def test_transformers_uses_sdpa_without_flash_attention(self):
        self._install_rex_module(_FakeWrapper)
        with mock.patch.object(rex_omni_backend, "_flash_attention_available", return_value=False):
            backend = rex_omni_backend.RexOmniBackend("IDEA-Research/Rex-Omni")
            self.assertEqual((True, "ok"), backend.load())

        options = _FakeWrapper.instances[0].kwargs
        self.assertEqual("transformers", options["backend"])
        self.assertEqual("sdpa", options["attn_implementation"])
        self.assertTrue(options["local_files_only"])

    def test_missing_package_and_model_construction_failure(self):
        backend = rex_omni_backend.RexOmniBackend("IDEA-Research/Rex-Omni")
        with mock.patch.object(
            rex_omni_backend,
            "_wrapper_class",
            side_effect=ImportError("No module named rex_omni"),
        ):
            ok, message = backend.load()
        self.assertFalse(ok)
        self.assertIn("package unavailable", message)
        self.assertIsNone(backend.wrapper)

        self._install_rex_module(_FailingWrapper)
        failing = rex_omni_backend.RexOmniBackend("IDEA-Research/Rex-Omni")
        ok, message = failing.load()
        self.assertFalse(ok)
        self.assertIn("failed to load", message)
        self.assertIn("fake model construction failed", message)
        self.assertIsNone(failing.wrapper)

    def test_missing_checkpoint_does_not_import_rex(self):
        backend = rex_omni_backend.RexOmniBackend("IDEA-Research/Rex-Omni")
        with mock.patch.object(rex_omni_backend, "resolve_hf", return_value=None), mock.patch.object(
            rex_omni_backend, "_wrapper_class"
        ) as import_wrapper:
            ok, message = backend.load()
        self.assertFalse(ok)
        self.assertIn("weights needed", message)
        import_wrapper.assert_not_called()


if __name__ == "__main__":
    unittest.main()

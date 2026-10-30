import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "qwen_asr"
    / "core"
    / "transformers_backend"
    / "configuration_qwen3_asr.py"
)
_SPEC = spec_from_file_location("qwen3_asr_test_configuration", _CONFIG_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_CONFIG_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONFIG_MODULE)
Qwen3ASRConfig = _CONFIG_MODULE.Qwen3ASRConfig


class AttentionImplementationPropagationTest(unittest.TestCase):
    def test_top_level_attention_implementation_reaches_nested_configs(self):
        config = Qwen3ASRConfig(attn_implementation="eager")

        self.assertEqual(config._attn_implementation, "eager")
        self.assertEqual(config.thinker_config._attn_implementation, "eager")
        self.assertEqual(config.thinker_config.audio_config._attn_implementation, "eager")
        self.assertEqual(config.thinker_config.text_config._attn_implementation, "eager")

    def test_nested_attention_implementation_is_preserved_without_top_level_override(self):
        config = Qwen3ASRConfig(thinker_config={"attn_implementation": "eager"})

        self.assertEqual(config.thinker_config._attn_implementation, "eager")
        self.assertEqual(config.thinker_config.audio_config._attn_implementation, "eager")
        self.assertEqual(config.thinker_config.text_config._attn_implementation, "eager")

    def test_attention_implementation_mapping_reaches_nested_configs(self):
        config = Qwen3ASRConfig(
            attn_implementation={"": "sdpa", "thinker_config": "eager"}
        )

        self.assertEqual(config._attn_implementation, "sdpa")
        self.assertEqual(config.thinker_config._attn_implementation, "eager")
        self.assertEqual(config.thinker_config.audio_config._attn_implementation, "eager")
        self.assertEqual(config.thinker_config.text_config._attn_implementation, "eager")


if __name__ == "__main__":
    unittest.main()

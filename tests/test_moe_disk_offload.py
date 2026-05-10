import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

from mlx_lm.models.moe_disk_offload import (
    DiskBackedSwitchGLU,
    SafetensorsExpertStore,
    auto_disk_moe_layers,
    is_expert_tensor_name,
    layers_from_last_n,
    parse_layer_spec,
    resolve_moe_offload_layers,
)
from mlx_lm.models.deepseek_v4 import Model as DeepseekV4Model
from mlx_lm.models.deepseek_v4 import ModelArgs as DeepseekV4ModelArgs
from mlx_lm.utils import load_model


PREFIX = "model.layers.0.ffn.switch_mlp"


def _make_indexed_safetensors(root: Path, tensors: dict):
    shard = root / "model.safetensors"
    mx.save_safetensors(str(shard), tensors, metadata={"format": "mlx"})
    with open(root / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": {k: shard.name for k in tensors}}, f)


class MoeDiskOffloadTests(unittest.TestCase):
    def test_axis0_expert_slice_matches_full_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key = f"{PREFIX}.up_proj.weight"
            full = mx.arange(24, dtype=mx.uint32).reshape(3, 4, 2)
            _make_indexed_safetensors(root, {key: full})

            store = SafetensorsExpertStore(root, cache_bytes=1024)
            sliced = store.read_expert_tensor(key, 1)

            self.assertEqual(sliced.shape, (1, 4, 2))
            self.assertTrue(mx.array_equal(sliced, full[1:2]).item())

    def test_deepseek_v4_raw_expert_tensors_are_layer_filterable(self):
        layers = parse_layer_spec("41-42")
        self.assertTrue(
            is_expert_tensor_name("layers.42.ffn.experts.w1.weight", layers=layers)
        )
        self.assertTrue(
            is_expert_tensor_name(
                "model.layers.42.ffn.switch_mlp.gate_proj.scales", layers=layers
            )
        )
        self.assertFalse(
            is_expert_tensor_name(
                "layers.42.ffn.shared_experts.w1.weight", layers=layers
            )
        )
        self.assertFalse(
            is_expert_tensor_name("layers.40.ffn.experts.w2.weight", layers=layers)
        )

    def test_last_n_disk_moe_layer_selection(self):
        self.assertEqual(layers_from_last_n(43, 20), set(range(23, 43)))
        self.assertEqual(layers_from_last_n(43, 0), set())
        self.assertEqual(layers_from_last_n(43, 99), set(range(43)))
        self.assertEqual(
            resolve_moe_offload_layers(
                {"num_hidden_layers": 43},
                explicit_layers="0-3",
                n_disk_moe=20,
            ),
            set(range(23, 43)),
        )

    def test_auto_disk_moe_keeps_prefix_layers_that_fit_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tensors = {
                "model.embed_tokens.weight": mx.ones((10,), dtype=mx.uint32),
                "layers.0.ffn.experts.w1.weight": mx.ones((10,), dtype=mx.uint32),
                "layers.1.ffn.experts.w1.weight": mx.ones((20,), dtype=mx.uint32),
                "layers.2.ffn.experts.w1.weight": mx.ones((30,), dtype=mx.uint32),
                "layers.3.ffn.experts.w1.weight": mx.ones((40,), dtype=mx.uint32),
            }
            _make_indexed_safetensors(root, tensors)

            # 40 bytes non-expert + 40 + 80 bytes for layers 0 and 1 fit.
            # Layer 2 would require another 120 bytes, so layers 2 and 3
            # become the disk-offloaded suffix.
            layers = auto_disk_moe_layers(
                {"num_hidden_layers": 4},
                root,
                available_memory_bytes=40 + 40 + 80 + 16,
                reserve_mb=0,
                cache_mb=0,
            )

            self.assertEqual(layers, {2, 3})

    def test_auto_disk_moe_respects_reserved_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tensors = {
                "model.embed_tokens.weight": mx.ones((10,), dtype=mx.uint32),
                "layers.0.ffn.experts.w1.weight": mx.ones((10,), dtype=mx.uint32),
                "layers.1.ffn.experts.w1.weight": mx.ones((10,), dtype=mx.uint32),
            }
            _make_indexed_safetensors(root, tensors)

            layers = auto_disk_moe_layers(
                {"num_hidden_layers": 2},
                root,
                available_memory_bytes=120,
                reserve_mb=1,
                cache_mb=0,
            )

            self.assertEqual(layers, {0, 1})

    def test_store_indexes_only_selected_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tensors = {
                "layers.40.ffn.experts.w1.weight": mx.ones((2, 4, 2)),
                "layers.42.ffn.experts.w1.weight": mx.ones((2, 4, 2)),
            }
            _make_indexed_safetensors(root, tensors)

            store = SafetensorsExpertStore(root, cache_bytes=4096, layers={42})

            self.assertFalse(store.has_tensor("layers.40.ffn.experts.w1.weight"))
            self.assertTrue(store.has_tensor("layers.42.ffn.experts.w1.weight"))

    def test_disk_backed_switch_glu_supports_deepseek_projection_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tensors = {}
            tensors["layers.0.ffn.experts.w1.weight"] = mx.ones((3, 4, 2))
            tensors["layers.0.ffn.experts.w3.weight"] = mx.ones((3, 4, 2))
            tensors["layers.0.ffn.experts.w2.weight"] = mx.ones((3, 2, 4))
            _make_indexed_safetensors(root, tensors)

            store = SafetensorsExpertStore(root, cache_bytes=4096)
            layer = DiskBackedSwitchGLU(
                2,
                4,
                3,
                "layers.0.ffn.experts",
                store=store,
                projection_names={
                    "gate_proj": "w1",
                    "up_proj": "w3",
                    "down_proj": "w2",
                },
            )

            x = mx.ones((1, 2, 2))
            indices = mx.array([[[2, 0], [0, 2]]], dtype=mx.int32)
            y = layer(x, indices)

            self.assertEqual(y.shape, (1, 2, 2, 2))
            self.assertEqual(store.stats["misses"], 6)

    def test_load_model_uses_last_n_disk_moe_for_deepseek_v4_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                "model_type": "deepseek_v4",
                "vocab_size": 16,
                "hidden_size": 8,
                "intermediate_size": 16,
                "moe_intermediate_size": 4,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "n_shared_experts": 1,
                "n_routed_experts": 2,
                "num_experts_per_tok": 1,
                "q_lora_rank": 4,
                "qk_rope_head_dim": 2,
                "head_dim": 4,
                "compress_ratios": [0, 0],
                "o_groups": 1,
                "o_lora_rank": 4,
                "index_n_heads": 1,
                "index_head_dim": 4,
                "index_topk": 2,
                "num_nextn_predict_layers": 0,
            }
            (root / "config.json").write_text(json.dumps(config))
            tensors = {
                "layers.1.ffn.experts.w1.weight": mx.ones((2, 4, 8)),
                "layers.1.ffn.experts.w2.weight": mx.ones((2, 8, 4)),
                "layers.1.ffn.experts.w3.weight": mx.ones((2, 4, 8)),
            }
            _make_indexed_safetensors(root, tensors)

            model, _ = load_model(root, lazy=True, strict=False, n_disk_moe=1)

            self.assertNotIsInstance(model.model.layers[0].ffn.switch_mlp, DiskBackedSwitchGLU)
            self.assertIsInstance(model.model.layers[1].ffn.switch_mlp, DiskBackedSwitchGLU)

    def test_deepseek_v4_sanitize_remaps_quantized_top_and_split_woa(self):
        args = DeepseekV4ModelArgs(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            moe_intermediate_size=4,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            n_routed_experts=2,
            num_experts_per_tok=1,
            q_lora_rank=4,
            qk_rope_head_dim=2,
            head_dim=4,
            compress_ratios=[0],
            o_groups=2,
            o_lora_rank=4,
            num_nextn_predict_layers=0,
        )
        model = DeepseekV4Model(args)
        weights = {
            "embed.weight": mx.ones((16, 1), dtype=mx.uint32),
            "embed.scales": mx.ones((16, 1)),
            "embed.biases": mx.ones((16, 1)),
            "head.weight": mx.ones((16, 1), dtype=mx.uint32),
            "head.scales": mx.ones((16, 1)),
            "head.biases": mx.ones((16, 1)),
            "layers.0.attn.wo_a.0.weight": mx.ones((4, 4)),
            "layers.0.attn.wo_a.1.weight": mx.ones((4, 4)) * 2,
            "layers.0.attn.wo_a.0.scales": mx.ones((4, 1)),
            "layers.0.attn.wo_a.1.scales": mx.ones((4, 1)) * 2,
            "layers.0.attn.wo_a.0.biases": mx.ones((4, 1)),
            "layers.0.attn.wo_a.1.biases": mx.ones((4, 1)) * 2,
        }

        sanitized = model.sanitize(weights)

        self.assertIn("model.embed_tokens.scales", sanitized)
        self.assertIn("lm_head.biases", sanitized)
        self.assertEqual(sanitized["model.layers.0.attn.wo_a.weight"].shape, (2, 4, 4))
        self.assertEqual(sanitized["model.layers.0.attn.wo_a.scales"].shape, (2, 4, 1))

    def test_deepseek_v4_sanitize_remaps_stacked_raw_experts(self):
        args = DeepseekV4ModelArgs(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            moe_intermediate_size=4,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            n_routed_experts=2,
            num_experts_per_tok=1,
            q_lora_rank=4,
            qk_rope_head_dim=2,
            head_dim=4,
            compress_ratios=[0],
            o_groups=1,
            o_lora_rank=4,
            num_nextn_predict_layers=0,
        )
        model = DeepseekV4Model(args)
        weights = {
            "layers.0.ffn.experts.w1.weight": mx.ones((2, 4, 8), dtype=mx.uint32),
            "layers.0.ffn.experts.w1.scales": mx.ones((2, 4, 1)),
            "layers.0.ffn.experts.w1.biases": mx.ones((2, 4, 1)),
            "layers.0.ffn.experts.w2.weight": mx.ones((2, 8, 4), dtype=mx.uint32),
            "layers.0.ffn.experts.w3.weight": mx.ones((2, 4, 8), dtype=mx.uint32),
        }

        sanitized = model.sanitize(weights)

        self.assertIn("model.layers.0.ffn.switch_mlp.gate_proj.weight", sanitized)
        self.assertIn("model.layers.0.ffn.switch_mlp.gate_proj.scales", sanitized)
        self.assertIn("model.layers.0.ffn.switch_mlp.gate_proj.biases", sanitized)
        self.assertIn("model.layers.0.ffn.switch_mlp.down_proj.weight", sanitized)
        self.assertIn("model.layers.0.ffn.switch_mlp.up_proj.weight", sanitized)
        self.assertNotIn("model.layers.0.ffn.experts.w1.weight", sanitized)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

from mlx_lm.models.moe_disk_offload import (
    DiskBackedSwitchGLU,
    SafetensorsExpertStore,
    is_expert_tensor_name,
    layers_from_last_n,
    parse_layer_spec,
    resolve_moe_offload_layers,
)


PREFIX = "language_model.model.layers.0.mlp.switch_mlp"


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

    def test_lru_cache_evicts_expert_slices_over_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key = f"{PREFIX}.up_proj.weight"
            full = mx.arange(48, dtype=mx.uint32).reshape(3, 4, 4)
            _make_indexed_safetensors(root, {key: full})

            store = SafetensorsExpertStore(root, cache_bytes=32)
            store.read_expert_tensor(key, 0)
            store.read_expert_tensor(key, 1)

            self.assertEqual(store.stats["misses"], 2)
            self.assertLessEqual(store.cached_bytes, 32)

    def test_layer_spec_filters_expert_tensors(self):
        layers = parse_layer_spec("1,3-4")
        self.assertEqual(layers, {1, 3, 4})
        self.assertFalse(is_expert_tensor_name(f"{PREFIX}.up_proj.weight", layers=layers))
        self.assertTrue(
            is_expert_tensor_name(
                "language_model.model.layers.1.mlp.switch_mlp.up_proj.weight",
                layers=layers,
            )
        )
        self.assertTrue(
            is_expert_tensor_name(
                "language_model.model.layers.4.mlp.switch_mlp.down_proj.scales",
                layers=layers,
            )
        )

    def test_last_n_disk_moe_layer_selection(self):
        self.assertEqual(layers_from_last_n(40, 20), set(range(20, 40)))
        self.assertEqual(layers_from_last_n(40, 0), set())
        self.assertEqual(layers_from_last_n(40, 99), set(range(40)))
        self.assertEqual(
            resolve_moe_offload_layers(
                {"text_config": {"num_hidden_layers": 40}},
                explicit_layers="0-3",
                n_disk_moe=20,
            ),
            set(range(20, 40)),
        )

    def test_store_indexes_only_selected_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tensors = {
                "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight": mx.ones(
                    (2, 4, 2)
                ),
                "language_model.model.layers.1.mlp.switch_mlp.up_proj.weight": mx.ones(
                    (2, 4, 2)
                ),
            }
            _make_indexed_safetensors(root, tensors)

            store = SafetensorsExpertStore(root, cache_bytes=4096, layers={1})

            self.assertFalse(
                store.has_tensor(
                    "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight"
                )
            )
            self.assertTrue(
                store.has_tensor(
                    "language_model.model.layers.1.mlp.switch_mlp.up_proj.weight"
                )
            )

    def test_disk_backed_switch_glu_remaps_global_expert_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tensors = {}
            for proj in ("gate_proj", "up_proj"):
                tensors[f"{PREFIX}.{proj}.weight"] = mx.ones((3, 4, 2))
            tensors[f"{PREFIX}.down_proj.weight"] = mx.ones((3, 2, 4))
            _make_indexed_safetensors(root, tensors)

            store = SafetensorsExpertStore(root, cache_bytes=4096)
            layer = DiskBackedSwitchGLU(2, 4, 3, PREFIX, store=store)

            x = mx.ones((1, 2, 2))
            indices = mx.array([[[2, 0], [0, 2]]], dtype=mx.int32)
            y = layer(x, indices)

            self.assertEqual(y.shape, (1, 2, 2, 2))
            self.assertEqual(store.stats["misses"], 6)


if __name__ == "__main__":
    unittest.main()

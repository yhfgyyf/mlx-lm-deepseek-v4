import json
import logging
import re
import subprocess
from dataclasses import dataclass
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .activations import swiglu


_CONFIG = {
    "mode": "none",
    "model_path": None,
    "cache_mb": 4096,
    "prefetch": False,
    "layers": None,
    "store": None,
}


def _numel(shape):
    n = 1
    for dim in shape:
        n *= int(dim)
    return n


def _dtype_info(dtype: str):
    if dtype == "BF16":
        return np.dtype("<u2"), mx.uint16, mx.bfloat16
    if dtype == "F8_E8M0":
        return np.dtype("u1"), mx.uint8, None
    mapping = {
        "F16": (np.dtype("<f2"), mx.float16, None),
        "F32": (np.dtype("<f4"), mx.float32, None),
        "I32": (np.dtype("<i4"), mx.int32, None),
        "I64": (np.dtype("<i8"), mx.int64, None),
        "U8": (np.dtype("u1"), mx.uint8, None),
        "U32": (np.dtype("<u4"), mx.uint32, None),
        "BOOL": (np.dtype("?"), mx.bool_, None),
    }
    return mapping.get(dtype)


def _read_header(path: Path):
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def _tensor_nbytes(entry: dict) -> int:
    start, end = [int(x) for x in entry["data_offsets"]]
    return end - start


def _iter_tensor_entries(model_path: str | Path):
    model_path = Path(model_path).expanduser()
    index_path = model_path / "model.safetensors.index.json"
    headers = {}

    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f).get("weight_map", {})
        for tensor_name, shard_name in weight_map.items():
            shard_path = model_path / shard_name
            if shard_path not in headers:
                headers[shard_path] = _read_header(shard_path)[0]
            entry = headers[shard_path].get(tensor_name)
            if entry is not None:
                yield tensor_name, entry
        return

    for shard_path in sorted(model_path.glob("*.safetensors")):
        header, _ = _read_header(shard_path)
        for tensor_name, entry in header.items():
            if tensor_name != "__metadata__":
                yield tensor_name, entry


def _array_from_bytes(raw, entry: dict):
    dtype_info = _dtype_info(str(entry["dtype"]))
    if dtype_info is None:
        raise TypeError(f"Unsupported safetensors dtype: {entry['dtype']}")
    np_dtype, mlx_dtype, bitcast_to = dtype_info
    shape = tuple(int(x) for x in entry["shape"])
    arr = np.frombuffer(raw, dtype=np_dtype, count=_numel(shape)).reshape(shape)
    out = mx.array(arr, dtype=mlx_dtype)
    if bitcast_to is not None:
        out = out.view(bitcast_to)
    return out


_LAYER_RE = re.compile(
    r"(?:^|\.)layers\.(\d+)\.(?:mlp|ffn)\.(?:switch_mlp|experts)\."
)
_SWITCH_EXPERT_RE = re.compile(
    r"(?:^|\.)layers\.\d+\.(?:mlp|ffn)\.switch_mlp\."
    r"(?:gate_proj|up_proj|down_proj)\.(?:weight|scales|biases|bias)$"
)
_DEEPSEEK_RAW_EXPERT_RE = re.compile(
    r"(?:^|\.)layers\.\d+\.ffn\.experts\.(?:w1|w2|w3)\."
    r"(?:weight|scales|biases|bias)$"
)


def parse_layer_spec(spec) -> Optional[set[int]]:
    if spec is None:
        return None
    if isinstance(spec, set):
        return {int(x) for x in spec}
    if isinstance(spec, (list, tuple)):
        return {int(x) for x in spec}
    value = str(spec).strip().lower()
    if value in {"", "all", "*"}:
        return None
    if value in {"none", "off", "false"}:
        return set()

    layers = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid layer range: {part}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    return layers


def _config_get(config: Any, key: str, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _num_hidden_layers_from_config(config: Any) -> Optional[int]:
    for candidate in (
        config,
        _config_get(config, "text_config"),
        _config_get(config, "language_config"),
    ):
        if candidate is None:
            continue
        value = _config_get(candidate, "num_hidden_layers")
        if value is not None:
            return int(value)
    return None


def available_memory_bytes_from_vm_stat(output: str, inactive_ratio: float = 0.5) -> int:
    page_size_match = re.search(r"page size of (\d+) bytes", output)
    page_size = int(page_size_match.group(1)) if page_size_match else 4096
    inactive_ratio = min(1.0, max(0.0, float(inactive_ratio)))
    page_weights = {
        "Pages free": 1.0,
        "Pages inactive": inactive_ratio,
        "Pages speculative": 1.0,
        "Pages purgeable": 1.0,
    }
    reclaimable_pages = 0.0
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().strip('"')
        weight = page_weights.get(key)
        if weight is None:
            continue
        match = re.search(r"\d+", value)
        if match:
            reclaimable_pages += int(match.group(0)) * weight
    return int(reclaimable_pages * page_size)


def available_memory_bytes(inactive_ratio: float = 0.5) -> int:
    try:
        import psutil

        memory = psutil.virtual_memory()
        inactive = getattr(memory, "inactive", 0)
        available = int(memory.available) - inactive
        available += int(inactive * min(1.0, max(0.0, float(inactive_ratio))))
        return max(0, available)
    except Exception:
        pass

    try:
        output = subprocess.check_output(["vm_stat"], text=True)
    except Exception as exc:
        raise RuntimeError("Could not determine available system memory") from exc

    return available_memory_bytes_from_vm_stat(output, inactive_ratio=inactive_ratio)


def layers_from_last_n(num_layers: int, n: int) -> set[int]:
    num_layers = int(num_layers)
    n = int(n or 0)
    if num_layers < 0:
        raise ValueError("num_layers must be non-negative")
    if n <= 0 or num_layers == 0:
        return set()
    n = min(n, num_layers)
    return set(range(num_layers - n, num_layers))


@dataclass(frozen=True)
class AutoDiskMoePlan:
    offload_layers: set[int]
    resident_layers: set[int]
    available_bytes: int
    resident_non_expert_bytes: int
    system_reserve_bytes: int
    runtime_reserve_bytes: int
    resident_expert_budget_bytes: int
    resident_expert_bytes: int
    estimated_resident_expert_bytes: int
    inactive_memory_ratio: float
    total_weight_bytes: int
    total_expert_bytes: int


def moe_expert_bytes_by_layer(model_path: str | Path) -> tuple[int, dict[int, int]]:
    total_bytes = 0
    expert_bytes = {}
    for tensor_name, entry in _iter_tensor_entries(model_path):
        nbytes = _tensor_nbytes(entry)
        total_bytes += nbytes
        if is_expert_tensor_name(tensor_name):
            layer_id = _layer_id_from_tensor_name(tensor_name)
            if layer_id is not None:
                expert_bytes[layer_id] = expert_bytes.get(layer_id, 0) + nbytes
    return total_bytes, expert_bytes


def _mb_to_bytes(value: int) -> int:
    return max(0, int(value)) * 1024 * 1024


def plan_auto_disk_moe_layers(
    config: Any,
    model_path: str | Path,
    available_memory_bytes: Optional[int] = None,
    reserve_mb: int = 4096,
    runtime_reserve_mb: int = 2048,
    cache_mb: int = 4096,
    resident_cost_scale: float = 1.0,
    inactive_memory_ratio: float = 0.5,
) -> AutoDiskMoePlan:
    num_layers = _num_hidden_layers_from_config(config)
    if num_layers is None:
        raise ValueError("--n-disk-moe auto requires num_hidden_layers in model config")

    if available_memory_bytes is None:
        available_memory_bytes = globals()["available_memory_bytes"](
            inactive_ratio=inactive_memory_ratio
        )

    total_bytes, expert_by_layer = moe_expert_bytes_by_layer(model_path)
    resident_non_expert = total_bytes - sum(expert_by_layer.values())
    system_reserve = _mb_to_bytes(reserve_mb)
    runtime_reserve = _mb_to_bytes(runtime_reserve_mb)
    budget = int(available_memory_bytes) - system_reserve - runtime_reserve
    budget -= resident_non_expert

    resident_prefix_layers = 0
    used = 0
    estimated_used = 0
    resident_cost_scale = max(0.0, float(resident_cost_scale))
    for layer_id in range(num_layers):
        layer_bytes = expert_by_layer.get(layer_id, 0)
        estimated_layer_bytes = int(layer_bytes * resident_cost_scale)
        if estimated_used + estimated_layer_bytes > budget:
            break
        used += layer_bytes
        estimated_used += estimated_layer_bytes
        resident_prefix_layers = layer_id + 1

    resident_layers = set(range(resident_prefix_layers))
    offload_layers = set(range(resident_prefix_layers, num_layers))
    return AutoDiskMoePlan(
        offload_layers=offload_layers,
        resident_layers=resident_layers,
        available_bytes=int(available_memory_bytes),
        resident_non_expert_bytes=resident_non_expert,
        system_reserve_bytes=system_reserve,
        runtime_reserve_bytes=runtime_reserve,
        resident_expert_budget_bytes=max(0, budget),
        resident_expert_bytes=used,
        estimated_resident_expert_bytes=estimated_used,
        inactive_memory_ratio=float(inactive_memory_ratio),
        total_weight_bytes=total_bytes,
        total_expert_bytes=sum(expert_by_layer.values()),
    )


def auto_disk_moe_layers(
    config: Any,
    model_path: str | Path,
    available_memory_bytes: Optional[int] = None,
    reserve_mb: int = 4096,
    runtime_reserve_mb: int = 2048,
    cache_mb: int = 4096,
    resident_cost_scale: float = 1.0,
    inactive_memory_ratio: float = 0.5,
) -> set[int]:
    return plan_auto_disk_moe_layers(
        config,
        model_path,
        available_memory_bytes=available_memory_bytes,
        reserve_mb=reserve_mb,
        runtime_reserve_mb=runtime_reserve_mb,
        cache_mb=cache_mb,
        resident_cost_scale=resident_cost_scale,
        inactive_memory_ratio=inactive_memory_ratio,
    ).offload_layers


def expand_auto_disk_moe_layers_with_reclaimable(
    config: Any,
    model_path: str | Path,
    offload_layers: set[int],
    available_memory_bytes: Optional[int] = None,
    reserve_mb: int = 4096,
    runtime_reserve_mb: int = 2048,
    resident_cost_scale: float = 1.0,
    inactive_memory_ratio: float = 0.5,
) -> set[int]:
    num_layers = _num_hidden_layers_from_config(config)
    if num_layers is None:
        raise ValueError("--n-disk-moe auto requires num_hidden_layers in model config")

    if available_memory_bytes is None:
        available_memory_bytes = globals()["available_memory_bytes"](
            inactive_ratio=inactive_memory_ratio
        )

    _, expert_by_layer = moe_expert_bytes_by_layer(model_path)
    extra_budget = (
        int(available_memory_bytes)
        - _mb_to_bytes(reserve_mb)
        - _mb_to_bytes(runtime_reserve_mb)
    )
    if extra_budget <= 0:
        return set(offload_layers)

    expanded_offload = set(offload_layers)
    used = 0
    resident_cost_scale = max(0.0, float(resident_cost_scale))
    for layer_id in range(num_layers):
        if layer_id not in expanded_offload:
            continue
        layer_bytes = expert_by_layer.get(layer_id, 0)
        estimated_layer_bytes = int(layer_bytes * resident_cost_scale)
        if used + estimated_layer_bytes > extra_budget:
            break
        used += estimated_layer_bytes
        expanded_offload.remove(layer_id)
    return expanded_offload


def n_disk_moe_requests_offload(n_disk_moe) -> bool:
    if str(n_disk_moe).strip().lower() == "auto":
        return True
    return int(n_disk_moe or 0) > 0


def resolve_moe_offload_layers(
    config: Any,
    explicit_layers="all",
    n_disk_moe: int | str = 0,
    model_path: Optional[str | Path] = None,
    cache_mb: int = 4096,
    reserve_mb: int = 4096,
    runtime_reserve_mb: int = 2048,
    resident_cost_scale: float = 1.0,
    inactive_memory_ratio: float = 0.5,
) -> Optional[set[int]]:
    if str(n_disk_moe).strip().lower() == "auto":
        if model_path is None:
            raise ValueError("--n-disk-moe auto requires model_path")
        plan = plan_auto_disk_moe_layers(
            config,
            model_path,
            reserve_mb=reserve_mb,
            runtime_reserve_mb=runtime_reserve_mb,
            cache_mb=cache_mb,
            resident_cost_scale=resident_cost_scale,
            inactive_memory_ratio=inactive_memory_ratio,
        )
        logging.info(
            "--n-disk-moe auto: available=%.2f GiB, non_moe=%.2f GiB, "
            "system_reserve=%.2f GiB, runtime_reserve=%.2f GiB, "
            "resident_moe_budget=%.2f GiB, resident_moe=%.2f GiB, "
            "estimated_resident_moe=%.2f GiB, resident_cost_scale=%.2f, "
            "inactive_memory_ratio=%.2f, "
            "resident_layers=%d, offload_layers=%d",
            plan.available_bytes / 1024**3,
            plan.resident_non_expert_bytes / 1024**3,
            plan.system_reserve_bytes / 1024**3,
            plan.runtime_reserve_bytes / 1024**3,
            plan.resident_expert_budget_bytes / 1024**3,
            plan.resident_expert_bytes / 1024**3,
            plan.estimated_resident_expert_bytes / 1024**3,
            resident_cost_scale,
            inactive_memory_ratio,
            len(plan.resident_layers),
            len(plan.offload_layers),
        )
        return plan.offload_layers
    if int(n_disk_moe or 0) > 0:
        num_layers = _num_hidden_layers_from_config(config)
        if num_layers is None:
            raise ValueError("--n-disk-moe requires num_hidden_layers in model config")
        return layers_from_last_n(num_layers, int(n_disk_moe))
    return parse_layer_spec(explicit_layers)


def _layer_id_from_tensor_name(name: str) -> Optional[int]:
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def is_expert_tensor_name(name: str, layers=None) -> bool:
    is_expert = bool(
        _SWITCH_EXPERT_RE.search(name) or _DEEPSEEK_RAW_EXPERT_RE.search(name)
    )
    if not is_expert:
        return False
    parsed_layers = parse_layer_spec(layers)
    if parsed_layers is None:
        return True
    layer_id = _layer_id_from_tensor_name(name)
    return layer_id in parsed_layers


def configure_moe_expert_offload(
    mode: str = "none",
    model_path: Optional[str | Path] = None,
    cache_mb: int = 4096,
    prefetch: bool = False,
    layers="all",
):
    if mode not in {"none", "disk-lru"}:
        raise ValueError("moe expert offload mode must be 'none' or 'disk-lru'")
    _CONFIG["mode"] = mode
    _CONFIG["model_path"] = Path(model_path).expanduser() if model_path else None
    _CONFIG["cache_mb"] = int(cache_mb)
    _CONFIG["prefetch"] = bool(prefetch)
    _CONFIG["layers"] = parse_layer_spec(layers)
    _CONFIG["store"] = None
    if mode == "disk-lru":
        if _CONFIG["model_path"] is None:
            raise ValueError("model_path is required for disk-lru MoE expert offload")
        _CONFIG["store"] = SafetensorsExpertStore(
            _CONFIG["model_path"],
            cache_bytes=max(0, int(cache_mb)) * 1024 * 1024,
            layers=_CONFIG["layers"],
        )


def get_moe_expert_store():
    return _CONFIG.get("store") if _CONFIG.get("mode") == "disk-lru" else None


def load_safetensors_excluding(
    weight_files: list[str],
    skip: Callable[[str], bool],
) -> dict:
    weights = {}
    for wf in weight_files:
        path = Path(wf)
        header, data_start = _read_header(path)
        for name, entry in header.items():
            if name == "__metadata__" or skip(name):
                continue
            start, end = [int(x) for x in entry["data_offsets"]]
            with open(path, "rb") as f:
                f.seek(data_start + start)
                raw = f.read(end - start)
            if len(raw) != end - start:
                raise IOError(f"Could not read tensor {name} from {path}")
            weights[name] = _array_from_bytes(memoryview(raw), entry)
    return weights


class SafetensorsExpertStore:
    def __init__(
        self,
        model_path: str | Path,
        cache_bytes: int = 4 * 1024**3,
        layers=None,
    ):
        self.model_path = Path(model_path).expanduser()
        self.cache_bytes = int(cache_bytes)
        self.layers = parse_layer_spec(layers)
        self._headers = {}
        self._tensor_locations = {}
        self._cache = OrderedDict()
        self._cached_bytes = 0
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}
        self._build_index()

    @property
    def cached_bytes(self):
        return self._cached_bytes

    def _build_index(self):
        index_path = self.model_path / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                weight_map = json.load(f).get("weight_map", {})
        else:
            weight_map = {
                name: p.name
                for p in sorted(self.model_path.glob("*.safetensors"))
                for name in _read_header(p)[0]
                if name != "__metadata__"
            }

        for tensor_name, shard_name in weight_map.items():
            if not is_expert_tensor_name(tensor_name, layers=self.layers):
                continue
            shard_path = self.model_path / shard_name
            if shard_path not in self._headers:
                self._headers[shard_path] = _read_header(shard_path)
            header, data_start = self._headers[shard_path]
            if tensor_name in header:
                self._tensor_locations[tensor_name] = (
                    shard_path,
                    data_start,
                    header[tensor_name],
                )

    def has_tensor(self, tensor_name: str) -> bool:
        return tensor_name in self._tensor_locations

    def projection_is_quantized(self, prefix: str) -> bool:
        return self.has_tensor(f"{prefix}.scales")

    def read_expert_tensor(self, tensor_name: str, expert_id: int):
        cache_key = (tensor_name, int(expert_id))
        if cache_key in self._cache:
            self.stats["hits"] += 1
            arr, size = self._cache.pop(cache_key)
            self._cache[cache_key] = (arr, size)
            return arr

        self.stats["misses"] += 1
        arr = self._read_axis0_slice(tensor_name, int(expert_id), int(expert_id) + 1)
        size = int(getattr(arr, "nbytes", arr.size * 4))
        if self.cache_bytes > 0 and size <= self.cache_bytes:
            self._cache[cache_key] = (arr, size)
            self._cached_bytes += size
            self._evict_to_budget()
        return arr

    def _evict_to_budget(self):
        while self._cached_bytes > self.cache_bytes and self._cache:
            _, (_, size) = self._cache.popitem(last=False)
            self._cached_bytes -= size
            self.stats["evictions"] += 1

    def _read_axis0_slice(self, tensor_name: str, axis0_start: int, axis0_end: int):
        if tensor_name not in self._tensor_locations:
            raise KeyError(f"Offloaded expert tensor not found: {tensor_name}")
        path, data_start, entry = self._tensor_locations[tensor_name]
        shape = tuple(int(x) for x in entry["shape"])
        if not shape:
            raise ValueError(f"Tensor {tensor_name} has no axis 0")
        if axis0_start < 0 or axis0_end > shape[0] or axis0_start >= axis0_end:
            raise IndexError(f"Expert slice {axis0_start}:{axis0_end} out of range")

        dtype_info = _dtype_info(str(entry["dtype"]))
        if dtype_info is None:
            raise TypeError(f"Unsupported safetensors dtype: {entry['dtype']}")
        np_dtype, _, _ = dtype_info
        row_bytes = _numel(shape[1:]) * np_dtype.itemsize
        start, _ = [int(x) for x in entry["data_offsets"]]
        byte_start = start + axis0_start * row_bytes
        byte_end = start + axis0_end * row_bytes
        with open(path, "rb") as f:
            f.seek(data_start + byte_start)
            raw = f.read(byte_end - byte_start)
        if len(raw) != byte_end - byte_start:
            raise IOError(f"Could not read tensor slice {tensor_name}[{axis0_start}:{axis0_end}]")
        sliced_entry = dict(entry)
        sliced_entry["shape"] = [axis0_end - axis0_start, *shape[1:]]
        sliced_entry["data_offsets"] = [0, byte_end - byte_start]
        return _array_from_bytes(memoryview(raw), sliced_entry)

    def load_projection(self, prefix: str, expert_ids: list[int]):
        tensors = {}
        for suffix in ("weight", "scales", "biases", "bias"):
            tensor_name = f"{prefix}.{suffix}"
            if not self.has_tensor(tensor_name):
                continue
            parts = [self.read_expert_tensor(tensor_name, expert_id) for expert_id in expert_ids]
            tensors[suffix] = mx.concatenate(parts, axis=0)
        if "weight" not in tensors:
            raise KeyError(f"Offloaded projection has no weight tensor: {prefix}")
        return tensors


class DiskBackedSwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        prefix: str,
        store: Optional[SafetensorsExpertStore] = None,
        bias: bool = False,
    ):
        super().__init__()
        self._input_dims = input_dims
        self._output_dims = output_dims
        self._num_experts = num_experts
        self.prefix = prefix
        self.store = store or get_moe_expert_store()
        self.bias = bias
        if self.store is None:
            raise ValueError("DiskBackedSwitchLinear requires a SafetensorsExpertStore")

    @property
    def input_dims(self):
        return self._input_dims

    @property
    def output_dims(self):
        return self._output_dims

    @property
    def num_experts(self):
        return self._num_experts

    def _local_indices(self, indices):
        mx.eval(indices)
        flat = indices.reshape(-1).tolist()
        expert_ids = sorted({int(x) for x in flat})
        remap = {expert_id: i for i, expert_id in enumerate(expert_ids)}
        local = [remap[int(x)] for x in flat]
        return expert_ids, mx.array(local, dtype=indices.dtype).reshape(indices.shape)

    def __call__(self, x, indices, sorted_indices=False):
        expert_ids, local_indices = self._local_indices(indices)
        tensors = self.store.load_projection(self.prefix, expert_ids)
        if "scales" in tensors:
            bits = max(1, int(tensors["weight"].shape[-1] * 32 // self.input_dims))
            group_size = max(1, int(self.input_dims // tensors["scales"].shape[-1]))
            x = mx.gather_qmm(
                x,
                tensors["weight"],
                tensors["scales"],
                tensors.get("biases"),
                rhs_indices=local_indices,
                transpose=True,
                group_size=group_size,
                bits=bits,
                mode="affine",
                sorted_indices=sorted_indices,
            )
        else:
            x = mx.gather_mm(
                x,
                tensors["weight"].swapaxes(-1, -2),
                rhs_indices=local_indices,
                sorted_indices=sorted_indices,
            )
        if "bias" in tensors:
            x = x + mx.expand_dims(tensors["bias"][local_indices], -2)
        return x


class DiskBackedSwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        prefix: str,
        store: Optional[SafetensorsExpertStore] = None,
        activation: Optional[Callable] = None,
        projection_names: Optional[dict[str, str]] = None,
        bias: bool = False,
    ):
        super().__init__()
        store = store or get_moe_expert_store()
        projection_names = projection_names or {
            "gate_proj": "gate_proj",
            "up_proj": "up_proj",
            "down_proj": "down_proj",
        }
        self.activation = activation or (lambda x_up, x_gate: swiglu(x_gate, x_up))
        self.gate_proj = DiskBackedSwitchLinear(
            input_dims,
            hidden_dims,
            num_experts,
            f"{prefix}.{projection_names['gate_proj']}",
            store,
            bias=bias,
        )
        self.up_proj = DiskBackedSwitchLinear(
            input_dims,
            hidden_dims,
            num_experts,
            f"{prefix}.{projection_names['up_proj']}",
            store,
            bias=bias,
        )
        self.down_proj = DiskBackedSwitchLinear(
            hidden_dims,
            input_dims,
            num_experts,
            f"{prefix}.{projection_names['down_proj']}",
            store,
            bias=bias,
        )

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            from .switch_layers import _gather_sort, _scatter_unsort

            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)

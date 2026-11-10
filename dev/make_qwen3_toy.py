"""Rewrite a small llama-architecture GGUF as a qwen3-architecture one, adding
random QK-norm weights, to test the qwen3 code paths (QK-norm, NEOX RoPE) of
the training graph against llama.cpp.

    python3 dev/make_qwen3_toy.py stories260K.gguf qwen3-toy.gguf
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "native", "llama.cpp", "gguf-py"))
import numpy as np  # noqa: E402
import gguf  # noqa: E402


def main(src: str, dst: str) -> None:
    reader = gguf.GGUFReader(src)
    writer = gguf.GGUFWriter(dst, "qwen3")
    n_layer = 0
    for field in reader.fields.values():
        name = field.name
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        new = "qwen3." + name[len("llama."):] if name.startswith("llama.") else name
        vtype = field.types[0]
        if vtype == gguf.GGUFValueType.ARRAY:
            sub = field.types[1]
            if sub == gguf.GGUFValueType.STRING:
                value = [bytes(field.parts[i]).decode("utf-8") for i in field.data]
            else:
                value = [field.parts[i].tolist()[0] for i in field.data]
            writer.add_array(new, value)
        elif vtype == gguf.GGUFValueType.STRING:
            writer.add_string(new, bytes(field.parts[field.data[0]]).decode("utf-8"))
        else:
            writer.add_key_value(new, field.parts[field.data[0]].tolist()[0], vtype)
        if name == "llama.block_count":
            n_layer = field.parts[field.data[0]].tolist()[0]
    emb = reader.get_field("llama.embedding_length").parts[-1].tolist()[0]
    heads = reader.get_field("llama.attention.head_count").parts[-1].tolist()[0]
    head_dim = emb // heads
    rng = np.random.default_rng(0)
    for tensor in reader.tensors:
        data = np.array(tensor.data)
        writer.add_tensor(tensor.name, data, raw_dtype=tensor.tensor_type)
    for layer in range(n_layer):
        for kind in ("q", "k"):
            weight = (1.0 + 0.3 * rng.standard_normal(head_dim)).astype(np.float32)
            writer.add_tensor(f"blk.{layer}.attn_{kind}_norm.weight", weight)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

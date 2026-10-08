"""Regenerate the tiny, synthetic ONNX embedding fixtures (no trained weights).

Optional maintainer step: install onnx==1.19.1, then run this file.
Normal npm tests use the committed JSON and need neither Python nor downloads.
Every graph accepts dynamic-length int32 text inputs and returns one Constant.
The real ONNX runtime therefore supplies the output's dtype, shape, and values.
"""
import base64
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper

cases = {
    "pooled": ([1, 8], list(range(1, 9)), TensorProto.FLOAT),
    "pooled-vector": ([8], list(range(1, 9)), TensorProto.FLOAT),
    "tokens": ([1, 4, 2], [1, 2, 3, 4, 5, 6, 100, 200], TensorProto.FLOAT),
    "zero-tokens": ([1, 4, 2], [0] * 8, TensorProto.FLOAT),
    "batched": ([2, 4], list(range(1, 9)), TensorProto.FLOAT),
    "rank-four": ([1, 1, 4, 2], list(range(1, 9)), TensorProto.FLOAT),
    "integer": ([1, 8], list(range(1, 9)), TensorProto.INT32),
}
fixtures = {}
for name, (shape, values, dtype) in cases.items():
    tensor = helper.make_tensor("value", dtype, shape, values)
    node = helper.make_node("Constant", [], ["embedding"], value=tensor)
    inputs = [helper.make_tensor_value_info(key, TensorProto.INT32, [1, "sequence"])
              for key in ("input_ids", "attention_mask")]
    output = helper.make_tensor_value_info("embedding", dtype, shape)
    graph = helper.make_graph([node], name, inputs, [output])
    model = helper.make_model(graph, producer_name="edgeinfer-test-fixtures",
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    fixtures[name] = {
        "shape": shape,
        "type": TensorProto.DataType.Name(dtype),
        "values": values,
        "base64": base64.b64encode(model.SerializeToString()).decode("ascii"),
    }
Path(__file__).with_name("embedding-models.json").write_text(
    json.dumps(fixtures, indent=2) + "\n"
)

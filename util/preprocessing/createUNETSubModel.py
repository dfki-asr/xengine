#!/usr/bin/env python3

import argparse
import onnx
from onnx import helper, checker, TensorProto
import numpy as np
from onnxhelper import Model


def createSubmodel(model, output):
    print("ONNX version: ", onnx.__version__)
    graph = model.graph
    org_g = graph.getOnnx()
    graph.name = "unet2D"
    initializer_subset = []
    inputs_subset = []
    outputs_subset = []
    nodes_subset = []

    # smaller
    #idx_start = 12
    #idx_end = 90
    #shape_in_out = [1, 60, 256, 256]

    # small
    #idx_start = 24
    #idx_end = 74
    #shape_in_out = [1, 240, 64, 64]

    # tiny
    idx_start = 30
    idx_end = 66
    shape_in_out = [1, 480, 32, 32]

    for idx, node in enumerate(graph.nodes):
        if node.name == "":
            node.name = createNodeName(node, idx, graph)
        if (idx >= idx_start and idx <= idx_end):
            nodes_subset.append(node.getOnnx())
            n_ins = [i for i in node.getOnnx().input]
            n_outs = [o for o in node.getOnnx().output]
            inputs_subset.extend([i for i in org_g.input if i.name in n_ins])
            outputs_subset.extend(
                [o for o in org_g.output if o.name in n_outs])
            initializer_subset.extend(
                [i for i in org_g.initializer if i.name in n_ins])
            print("PICK ", idx, ": ", node.name, ", type: ", node.op_type)
        else:
            print("DROP ", idx, ": ", node.name, ", type: ", node.op_type)
    new_g_input = nodes_subset[0].input[0]
    inputs_subset = [
        helper.make_tensor_value_info(new_g_input, onnx.TensorProto.FLOAT,
                                      shape_in_out), *inputs_subset
    ]
    new_g_output = nodes_subset[len(nodes_subset) - 1].output[0]
    outputs_subset.append(
        helper.make_tensor_value_info(new_g_output, onnx.TensorProto.FLOAT,
                                      shape_in_out))

    new_g = helper.make_graph(nodes_subset, graph.name, inputs_subset,
                              outputs_subset, initializer_subset)
    new_m = helper.make_model(new_g,
                              ir_version=model.ir_version,
                              producer_name="schuler")
    print("save model to ", output)
    onnx.checker.check_model(new_m)
    onnx.save(new_m, output)


def main():
    parser = argparse.ArgumentParser(description="Rename ONNX graph nodes.")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="The input ONNX model file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="The output ONNX model file.",
    )
    args = parser.parse_args()
    # Load model into intermediate representation
    model = Model(onnx.load(args.model))
    createSubmodel(model, args.output)


if __name__ == "__main__":
    main()

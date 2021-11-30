#!/usr/bin/env python3

import argparse
import onnx
from onnx import helper, checker, TensorProto
import numpy as np
from hpdlf_onnx import Model


def setNodeNames(model):
    """ Create names for all nodes in graph if not already there """
    def count_number_of_nodes_of_type(graph, op_type, idx=-1):
        return sum(op_type == n.op_type for n in graph.nodes[0:idx])

    def createNodeName(node, idx, graph):
        node_nr = count_number_of_nodes_of_type(graph, node.op_type, idx)
        return node.op_type.lower() + str(node_nr)

    graph = model.graph
    for idx, node in enumerate(graph.nodes):
        if node.name == "":
            node.name = createNodeName(node, idx, graph)
    return model


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
        "-b",
        "--batchsize",
        type=int,
        required=False,
        default=-1,
        help="The batchsize. Default: use batchsize in ONNX model.",
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
    # Set names and batchsize
    model = setNodeNames(model)
    if args.batchsize != -1:
        model.setBatchsize(args.batchsize)
    # Translate back to ONNX
    onnx_model = model.getOnnx()
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, args.output)


if __name__ == "__main__":
    main()

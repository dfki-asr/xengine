import onnx
import onnx.shape_inference
import onnx.numpy_helper
import numpy as np
from . import utils
from .tensorInfo import TensorInfo
from .decorators import CachedDictDecorator, ListDecorator, DictDecorator
from .node import Node
from .nonCopyable import NonCopyable


class ValueInfosWrapper(CachedDictDecorator):
    """Wrapper around ONNX ValueInfo repeated struct with dictionary like access"""
    def _getKey(self, value):
        return value.name

    def _fromOnnx(self, value):
        return value if isinstance(value,
                                   TensorInfo) else TensorInfo.fromOnnx(value)

    def _toOnnx(self, key, value):
        if isinstance(value, onnx.ValueInfoProto):
            return value
        assert isinstance(value, TensorInfo)
        assert value.name == key
        return value.toOnnx()


class NodesWrapper(ListDecorator):
    """Wrapper around ONNX nodes"""
    def __init__(self, onnxList, model):
        self.model = model
        super().__init__(onnxList)

    def _fromOnnx(self, value):
        return value if isinstance(value, Node) else Node(value, self.model)

    def _toOnnx(self, value):
        return value if isinstance(value, onnx.NodeProto) else value.getOnnx()


class TensorWrapper(DictDecorator):
    """Wrapper around ONNX tensors (initializers)"""
    def _getKey(self, value):
        return value.name

    def _fromOnnx(self, value):
        return value if isinstance(
            value, np.ndarray) else onnx.numpy_helper.to_array(value)

    def _toOnnx(self, key, value):
        if isinstance(value, onnx.TensorProto):
            return value
        assert isinstance(value, np.ndarray)
        return onnx.numpy_helper.from_array(value, name=key)


class Graph(NonCopyable):
    """Wrapper around an ONNX graph in a model"""
    def __init__(self, onnxGraph, model):
        self._onnxGraph = onnxGraph
        self._model = model
        self._inputs = ValueInfosWrapper(self._onnxGraph.input)
        self._outputs = ValueInfosWrapper(self._onnxGraph.output)
        self._value_infos = ValueInfosWrapper(self._onnxGraph.value_info)
        self._nodes = NodesWrapper(self._onnxGraph.node, self._model)
        self._initializers = TensorWrapper(self._onnxGraph.initializer)

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def value_infos(self):
        return self._value_infos

    @property
    def nodes(self):
        return self._nodes

    @property
    def initializers(self):
        return self._initializers

    def getOnnx(self):
        return self._onnxGraph

    def tensorExists(self, name):
        """Return wether a tensor with the given name exists"""
        return any(name in d
                   for d in (self.inputs, self.outputs, self.value_infos))

    def getTensor(self, name):
        """Get the TensorInfo by name"""
        for info in (self.inputs, self.outputs, self.value_infos):
            result = info.get(name, None)
            if result:
                return result
        raise KeyError("Tensor '{}' not found".format(name))

    def getTensorShape(self, tensorName, assertNumeric=True):
        """Get the shape of a tensor by its name

        tensorName -- Name of the tensor in the graph
        assertNumeric -- Raise exception if any value in the shape is not numeric
        Note: assertNumeric implies that the result is a numpy array
        """
        tensor = self.getTensor(tensorName)
        return tensor.getNumericShape() if assertNumeric else tensor.shape

    def calcTensorSize(self, tensorName, sizeInBytes=True):
        """Calculate the size of a tensor by its name.

        sizeInBytes -- If True, return size in bytes, else number of elements
        """
        return self.getTensor(tensorName).calcSize(sizeInBytes)

    def replaceTensor(self, tensor):
        """Replace a tensor in ONNX model
        """
        if not self.tensorExists(tensor.name):
            raise RuntimeError(
                "Cannot replace tensor with name {}: does not exist".format(
                    tensor.name))

        def do_reshape(onnxList):
            for i in onnxList:
                if i.name == tensor.name:
                    for j in range(0, len(tensor.shape) - 1):
                        i.type.tensor_type.shape.dim[
                            j].dim_value = tensor.shape[j]
                    break

        # Replace in value infos
        for name in ('input', 'output', 'value_info'):
            attr = getattr(self, name + 's')
            if tensor.name in attr:
                do_reshape(getattr(self.getOnnx(), name))
                attr.invalidate()
                break
        # Replace in initializers
        if tensor.name in self.initializers:
            do_reshape(self.getOnnx().initializer)
            self.initializers.invalidate()

    def renameTensor(self, oldName, newName):
        """Rename a tensor by its name

        Changes (potentially) the whole graph invalidating all references to the modified tensor
        and instances referring to it (e.g. nodes)
        """
        if oldName == newName:
            return
        if self.tensorExists(newName):
            raise RuntimeError(
                "Cannot rename {} to {} as a tensor with the new name does already exists"
                .format(oldName, newName))

        def do_rename(onnxList):
            for i in onnxList:
                if i.name == oldName:
                    i.name = newName
                    break

        # Replace in value infos
        for name in ('input', 'output', 'value_info'):
            attr = getattr(self, name + 's')
            if oldName in attr:
                do_rename(getattr(self.getOnnx(), name))
                attr.invalidate()
                break
        # Replace in initializers
        if oldName in self.initializers:
            do_rename(self.getOnnx().initializer)
            self.initializers.invalidate()
        # Replace in node input/output
        for node in self.nodes:
            node.inputs[:] = [
                newName if v == oldName else v for v in node.inputs
            ]
            node.outputs[:] = [
                newName if v == oldName else v for v in node.outputs
            ]

    def setBatchsize(self, batch_size):
        input_tensor = self.getTensor(self.nodes[0].inputs[0].name)
        input_tensor.shape = np.array(
            (batch_size, *input_tensor.shape[1:])).astype(np.uint64)
        tensor_names = set(
            (input_tensor.name, *self.value_infos, *self.outputs))
        for tensor in [self.getTensor(tensor) for tensor in tensor_names]:
            if len(tensor.shape) == 1:
                continue
            tensor.shape = np.array(
                (batch_size, *tensor.shape[1:])).astype(np.uint64)
            self.replaceTensor(tensor)


utils.wrapOnnxProperties(
    Graph, onnx.GraphProto,
    ('input', 'output', 'value_info', 'node', 'initializer'))

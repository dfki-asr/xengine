import onnx
from .decorators import ListDecorator, CachedDictDecorator
from .utils import wrapOnnxProperties
from .nonCopyable import NonCopyable


class TensorListDecorator(ListDecorator):
    """Wrap a list of tensor names to their tensor info or name if no tensor info is found"""

    def __init__(self, onnxList, model):
        self._model = model
        super().__init__(onnxList)

    def _fromOnnx(self, value):
        if not isinstance(value, str):
            raise NotImplementedError(
                "Handling anything but strings is not used")
        try:
            return self._model.graph.getTensor(value)
        except KeyError:
            return value

    def _toOnnx(self, value):
        # Accept either a TensorInfo or a string
        if isinstance(value, str):
            return value
        try:
            tensor = self._model.graph.getTensor(value.name)
        except KeyError:
            raise ValueError(
                "Cannot add tensor data through this. Use a plain string instead"
            )
        # Note: Testing same instance here not same value
        if tensor != value:
            raise ValueError(
                "Cannot change tensor data through this. Use a plain string instead"
            )
        return value.name


class AttributesWrapper(CachedDictDecorator):
    """Wrapper around ONNX attribute repeated struct with dictionary like access

    Important: Ignores doc_strings (deleted on write)
    """

    def _getKey(self, value):
        return value.name

    def _fromOnnx(self, value):
        if not isinstance(value, onnx.AttributeProto):
            return value
        if value.ref_attr_name:
            raise NotImplementedError(
                "ref_attr_name is currently not supported")
        return onnx.helper.get_attribute_value(value)

    def _toOnnx(self, key, value):
        return value if isinstance(
            value, onnx.AttributeProto) else onnx.helper.make_attribute(
                key, value)


class Node(NonCopyable):
    """Wrapper around an ONNX node (operation)"""

    def __init__(self, onnxNode, model):
        self._onnxNode = onnxNode
        self._model = model
        self._inputs = TensorListDecorator(onnxNode.input, self._model)
        self._outputs = TensorListDecorator(onnxNode.output, self._model)
        self._attributes = AttributesWrapper(onnxNode.attribute)

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def attributes(self):
        """Dictionary of attribute names to values"""
        return self._attributes

    def getOnnx(self):
        return self._onnxNode

    def getOpVersion(self):
        """Get the version of the operator used for this graph"""
        opsetVersion = self._model.opsetVersions[self.domain]
        return onnx.defs.get_schema(self.op_type, opsetVersion,
                                    self.domain).since_version

    def getCanonical(self):
        """Return a canonical representation of the node"""
        # ONNX defines the canonical string representation as domain.type:version
        formatString = "{domain}{opType}:{version}"
        if self.name:
            formatString = "{name}(" + formatString + ")"
        return formatString.format(name=self.name,
                                   domain=self.domain +
                                   "." if self.domain else "",
                                   opType=self.op_type,
                                   version=self.getOpVersion())


wrapOnnxProperties(Node, onnx.NodeProto, ('input', 'output', 'attribute'))

from copy import deepcopy
import onnx
import onnx.shape_inference
from .graph import Graph
from .decorators import CachedDictDecorator, StringStringDict
from .utils import wrapOnnxProperties


class OpsetDict(CachedDictDecorator):

    def _getKey(self, value):
        return value.domain

    def _fromOnnx(self, value):
        return value if isinstance(value, int) else value.version

    def _toOnnx(self, key, value):
        return value if isinstance(
            value, onnx.OperatorSetIdProto) else onnx.helper.make_opsetid(
                key, value)


class ModelError(RuntimeError):
    """Exception raised when the model is invalid"""


class Model:
    """Pythonic representation of an ONNX model

    An ONNX model is checked for correctness on construction.
    You should rerun this check after modifications with `validate`.
    """

    def __init__(self, onnxModel):
        self._model = None
        self._validate(onnxModel)
        self._initFromValidatedModel(
            onnx.shape_inference.infer_shapes(onnxModel))

    def _initFromValidatedModel(self, model):
        """Init from a model which has already run through validation and shape inference"""
        self._model = model
        self._opsetVersions = OpsetDict(self._model.opset_import)
        self._metaData = StringStringDict(self._model.metadata_props)
        if "" not in self.opsetVersions:
            raise ModelError("Default opset not found")
        if self.opsetVersions[""] > onnx.defs.onnx_opset_version():
            raise ModelError(
                "ONNX python module is too old. Required: {}. Found: {}.".
                format(self.opsetVersions[""], onnx.defs.onnx_opset_version()))
        self._graph = Graph(self._model.graph, self)

    def __deepcopy__(self, memo):
        cls = self.__class__
        rv = cls.__new__(cls)  # No __init__ called!
        rv._initFromValidatedModel(deepcopy(self._model, memo))  # pylint: disable=protected-access
        return rv

    def validate(self):
        """(Re-)Run model validation"""
        self._validate(self._model)

    @staticmethod
    def _validate(model):
        try:
            onnx.checker.check_model(model)
        except Exception as e:
            raise ModelError(str(e)) from e

    def inferShapes(self):
        """Run shape inference and return the new model"""
        return Model(onnx.shape_inference.infer_shapes(self._model))

    @property
    def opsetVersions(self):
        return self._opsetVersions

    @property
    def metaData(self):
        return self._metaData

    @property
    def graph(self):
        return self._graph

    def getOnnx(self):
        return self._model

    def setBatchsize(self, batch_size):
        self.graph.setBatchsize(batch_size)
        self.inferShapes()


wrapOnnxProperties(Model, onnx.ModelProto,
                   ('graph', 'opset_import', 'metadata_props'))

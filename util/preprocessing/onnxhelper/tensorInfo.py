import numpy as np
from onnx import helper as onnxHelper
from . import utils


class TensorInfo:
    """Information on a tensor

    elemType        -- (Python) Type of each element
    shape           -- Array containing the number of elements per dimension
                       Will be a numpy array if fully numeric, else a list
    doc_string      -- Optional documentation
    shapeDenotation -- Array containing the description of each shape index
    """

    def __init__(self,
                 name,
                 elemType,
                 shape,
                 doc_string=None,
                 shapeDenotation=None):
        self._name = name
        self.elemType = elemType
        self._shape = None
        self.shape = shape
        self.docString = doc_string
        self.shapeDenotation = shapeDenotation

    def getNumericShape(self):
        """Get the shape if it is numeric or raise Exception

        Note: implies that the result is a numpy array
        """
        shape = self.shape
        if isinstance(shape, np.ndarray):
            # Should only be a numpy array if it is numeric
            assert np.issubdtype(shape.dtype, np.uint64)
        else:
            raise Exception("Found variables in tensor shape:", shape)
        return shape

    def calcSize(self, sizeInBytes=True):
        """Calculate the size of this tensor

        sizeInBytes -- If True, return size in bytes, else number of elements
        Requires the shape to be numeric
        """
        numElements = int(self.getNumericShape().prod())
        return numElements * utils.getElementSize(
            self.elemType) if sizeInBytes else numElements

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        # if numeric numpy array or python type store as numpy array else as list
        if (isinstance(value, np.ndarray)
                and not np.issubdtype(value.dtype, np.floating)) or all(
                    isinstance(x, int) for x in value):
            assert all(v >= 0 for v in value)
            self._shape = np.array(value, dtype=np.uint64)
        else:
            self._shape = [v for v in value]

    @property
    def name(self):
        """Readonly name"""
        return self._name

    @staticmethod
    def fromOnnx(onnxValueInfo):
        tensor_type = onnxValueInfo.type.tensor_type
        shape = utils.onnxShape2Python(tensor_type.shape)
        shapeDenotation = [dim.denotation for dim in tensor_type.shape.dim]
        return TensorInfo(onnxValueInfo.name, tensor_type.elem_type, shape,
                          onnxValueInfo.doc_string, shapeDenotation)

    def toOnnx(self):
        return onnxHelper.make_tensor_value_info(self.name, self.elemType,
                                                 self.shape, self.docString,
                                                 self.shapeDenotation)

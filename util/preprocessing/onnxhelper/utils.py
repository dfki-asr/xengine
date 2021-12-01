from onnx import TensorProto


def onnxShape2Python(shape):
    """Convert a TensorShapeProto to a python list"""
    return [getattr(dim, dim.WhichOneof("value")) for dim in shape.dim]


def getElementSize(elementType):
    """Get the size of a tensor element type in bytes"""
    elementSizes = {
        TensorProto.FLOAT: 4,
        TensorProto.UINT8: 1,
        TensorProto.INT8: 1,
        TensorProto.UINT16: 2,
        TensorProto.INT16: 2,
        TensorProto.INT32: 4,
        TensorProto.INT64: 8,
        TensorProto.BOOL: 1,
        TensorProto.FLOAT16: 2,
        TensorProto.DOUBLE: 8,
        TensorProto.UINT32: 4,
        TensorProto.UINT64: 8,
        TensorProto.COMPLEX64: 8,
        TensorProto.COMPLEX128: 16,
    }
    if elementType in elementSizes:
        return elementSizes[elementType]
    if elementType == TensorProto.STRING:
        raise ValueError("Cannot compute tensor size of string tensors")
    if elementType == TensorProto.UNDEFINED:
        raise ValueError("Cannot use a tensor of undefined type")
    raise NotImplementedError(
        "Element type {} is not supported".format(elementType))


def wrapOnnxProperties(pythonType, onnxType, exclude=None):
    """Extend a python type by adding redirecting properties to an underlying ONNX property

    pythonType   -- Type to extend. Must implement `getOnnx` function to get onnxType instance
    onnxType     -- ONNX type that is wrapped
    exclude      -- List of field names not to wrap
    """
    for descriptor in onnxType.DESCRIPTOR.fields:
        name = descriptor.name
        if exclude and name in exclude:
            continue

        # Required to capture name
        def createProperty(name):
            return property(
                fget=lambda self: getattr(self.getOnnx(), name),
                fset=lambda self, value: setattr(self.getOnnx(), name, value),
                fdel=lambda self: self.getOnnx().ClearField(name))

        setattr(pythonType, name, createProperty(name))


def reassignOnnxList(onnxList, newList):
    """Assign a python list to an ONNX list"""
    # Can't reassign directly, so delete all and append
    del onnxList[:]
    onnxList.extend(newList)

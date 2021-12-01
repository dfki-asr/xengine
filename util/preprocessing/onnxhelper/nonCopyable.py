class NonCopyable:  # pylint: disable=too-few-public-methods
    """Base class for wrappers that prevent deepcopying them

    This applies to all classes containing a reference to an ONNX object except the Model itself.
    Reason is that this reference is inside the model and copying the object would decouple it
    from the model.
    """
    def __deepcopy__(self, memo):
        raise RuntimeError(
            "Cannot deepcopy {}. Use deepcopy on the Model instance instead".
            format(self.__class__.__name__))

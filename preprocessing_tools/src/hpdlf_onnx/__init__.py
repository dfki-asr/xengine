"""ONNX helpers for HPDLF
====================================

Provides convenience methods for working with ONNX models.
All complex ONNX types are wrapped to provide more pythonic usage.
All dictionary-like protobuf lists are accessible by a dictionary interface.

Usage is as simple as creating an instance of `Model` from an ONNX model and use this instead of
the original model.
While it is optimized for read-only access manipulation of the model is possible and reflected in
the underlying ONNX model, which can be queried by the `getOnnx` method.
IMPORTANT: Changes to non-trivial types are only reflected after assignment to their container.
Example: model.graph.inputs['foo'].docString = 'bar' # Does NOT change the value in ONNX
Better: foo = model.graph.inputs['foo']
        foo.docString = 'bar'           # Now changed in the wrapper only
        model.graph.inputs['foo'] = foo # Persist to ONNX

Usage:
------

  >>> # Create a graph from an ONNX model
  >>> model = hpdlf.onnx.Model(onnxModel)
  >>> model.metaData['foo']              # -> value of metadata_props
  >>> model.opsetVersions['domain']      # -> value of opset of 'domain'
  >>> assert mode.getOnnx() == onnxModel # OK
  >>> model.producer_name                # Raw ONNX data with r/w access
  >>> graph = model.graph
  >>> weights = graph.inputs['foo_weights']    # Same for outputs and value_infos
  >>> weights = graph.getTensor('foo_weights') # Get from either of above 3 (names are unique)
  >>> weights.shape                            # Numpy uint64-array if numeric, list otherwise
  >>> for node in graph.nodes:                 # List interface
  >>>   node.inputs                            # List of strings
  >>>   node.attributes['alpha']               # of type float, int, ...
  >>>   node.getOpVersion()                    # ONNX operator version (depends on model)

Notes:
------

Not all python types might be exactly representable in ONNX format hence data loss may occur
when storing a Python value into an ONNX field
(e.g. float (by default 64Bit value) when the underlying ONNX float field has only 32Bit).
Due to caching this fact might not arise immediately:

Example:
  >>> node.attributes['floatVal'] = floatVal           # Assign a Python float
  >>> assert node.attributes['floatVal'] == floatVal   # When value is cached, this holds
  >>> assert node.getOnnx().attribute[0].f == floatVal # May fail when floatVal cannot losslessly
                                                       # converted to 32Bit float
"""
from .version import version as __version__
from .model import Model, ModelError
from .tensorInfo import TensorInfo

__all__ = ['__version__', 'Model', 'ModelError', 'TensorInfo']

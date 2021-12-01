"""Wrappers around ONNX repeated structs with easier access (Sequence or Mapping interface)

Subclasses need to implement _fromOnnx and _toOnnx (+ _getKey for Mappings) to convert
from and to an ONNX value.
IMPORTANT: Both functions must be able to deal with beeing passed a converted value and
           return this unmodified
"""

from collections import MutableSequence, MutableMapping
from abc import abstractmethod
from onnx import StringStringEntryProto
from .utils import reassignOnnxList
from .nonCopyable import NonCopyable


class ListDecorator(MutableSequence, NonCopyable):
    """Abstract wrapper around ONNX repeated struct with list like access

    Subclasses need to implement _fromOnnx, _toOnnx
    """
    def __init__(self, onnxList):
        """Create an instance

        onnxList -- ONNX protobuf list
        """
        self._list = onnxList

    @abstractmethod
    def _fromOnnx(self, value):
        """Convert an ONNX value to a value to be stored in the list"""

    @abstractmethod
    def _toOnnx(self, value):
        """Convert a value to an ONNX value"""

    def __getitem__(self, key):
        result = self._list[key]
        return [self._fromOnnx(v) for v in result] if isinstance(
            key, slice) else self._fromOnnx(result)

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            convValue = [self._toOnnx(v) for v in value]
        else:
            convValue = self._toOnnx(value)
        # Convert to list to support slice handling
        new = list(self._list)
        new[key] = convValue
        reassignOnnxList(self._list, new)

    def __delitem__(self, key):
        del self._list[key]

    def __len__(self):
        return len(self._list)

    def insert(self, index, value):
        convValue = self._toOnnx(value)
        # Just create a new list
        new = [v for v in self._list]
        new.insert(index, convValue)
        del self._list[:]
        self._list.extend(new)


class DictDecorator(MutableMapping, NonCopyable):
    """Wrapper around ONNX repeated struct with dictionary like access

    Subclasses need to implement _getKey, _fromOnnx, _toOnnx
    """
    def __init__(self, onnxList):
        """Create an instance

        onnxList  -- ONNX protobuf list
        """
        self._list = onnxList
        self.invalidate()

    @abstractmethod
    def _getKey(self, value):
        """Convert an ONNX value to a value to be stored in the list"""

    @abstractmethod
    def _fromOnnx(self, value):
        """Convert an ONNX value to a value to be stored in the list"""

    @abstractmethod
    def _toOnnx(self, key, value):
        """Convert a value to an ONNX value"""

    def invalidate(self):
        """Reload data from the underlying onnx list"""
        # Nothing do do

    def _getIndex(self, key):
        """Return the index of the element with the given key or -1"""
        return next(
            (i for i, v in enumerate(self._list) if self._getKey(v) == key),
            -1)

    def __contains__(self, key):
        return self._getIndex(key) >= 0

    def __getitem__(self, key):
        idx = self._getIndex(key)
        if idx < 0:
            raise KeyError(key)
        return self._fromOnnx(self._list[idx])

    def __setitem__(self, key, value):
        convValue = self._toOnnx(key, value)
        convKey = self._getKey(convValue)
        if convKey != key:
            raise ValueError(
                "Name or key '{}' of passed value does not match used key '{}'"
                .format(convKey, key))
        idx = self._getIndex(key)
        if idx >= 0:
            # Can't reassign an element, so create a copy with the element replaced
            # and assign whole list
            new = [
                convValue if i == idx else v for i, v in enumerate(self._list)
            ]
            reassignOnnxList(self._list, new)
        else:
            self._list.extend([convValue])

    def __delitem__(self, key):
        idx = self._getIndex(key)
        if idx >= 0:
            del self._list[idx]

    def __iter__(self):
        return (self._getKey(v) for v in self._list)

    def __len__(self):
        return len(self._list)


class CachedDictDecorator(DictDecorator):
    """Wrapper around ONNX repeated struct with dictionary like access which caches items

    Subclasses need to implement _getKey, _fromOnnx, _toOnnx
    """

    # pylint: disable=W0223

    def invalidate(self):
        """Reload data from the underlying onnx list"""
        self._data = {self._getKey(v): self._fromOnnx(v) for v in self._list}

    def __contains__(self, key):
        return key in self._data

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._data[key] = self._fromOnnx(value)

    def __delitem__(self, key):
        super().__delitem__(key)
        del self._data[key]

    def __iter__(self):
        return iter(self._data)


class StringStringDict(CachedDictDecorator):
    """Decorator for cross-proto-version maps

    See https://developers.google.com/protocol-buffers/docs/proto3#maps
    Entries are of type StringStringEntryProto (key, value as members)
    """
    def _getKey(self, value):
        return value.key

    def _fromOnnx(self, value):
        return value if isinstance(value, str) else value.value

    def _toOnnx(self, key, value):
        if isinstance(value, StringStringEntryProto):
            assert value.key == key
            return value
        result = StringStringEntryProto()
        result.key = key
        result.value = value
        return result

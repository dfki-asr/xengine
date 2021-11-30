#ifndef XENGINE_OP_RELU_HPP
#define XENGINE_OP_RELU_HPP

#include "eltwise.hpp"

class Relu : public Eltwise {
public:
  Relu(string n, vector<string> i, vector<string> o,
       unordered_map<string, unique_ptr<Tensor>> &tensors, int training)
      : Eltwise(n, "Relu", i, o, 0.0f, tensors, training) {}
};
#endif
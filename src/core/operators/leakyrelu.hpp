#ifndef XENGINE_OP_LEAKYRELU_HPP
#define XENGINE_OP_LEAKYRELU_HPP

#include "eltwise.hpp"

class LeakyRelu : public Eltwise {
public:
  LeakyRelu(string n, vector<string> i, vector<string> o, float a,
            unordered_map<string, shared_ptr<Tensor>> &tensors, int training)
      : Eltwise(n, "LeakyRelu", i, o, a, tensors, training) {}
};
#endif
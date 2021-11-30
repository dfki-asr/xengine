#ifndef XENGINE_OP_MAXPOOL_HPP
#define XENGINE_OP_MAXPOOL_HPP

#include "pool.hpp"

class MaxPool : public Pool {
public:
  MaxPool(string n, vector<string> i, vector<string> o, memory::dims s,
          memory::dims k, memory::dims p,
          unordered_map<string, unique_ptr<Tensor>> &tensors, int training)
      : Pool(n, "MaxPool", i, o, s, k, p, tensors, training) {}
};
#endif
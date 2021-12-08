#ifndef XENGINE_OP_AVGPOOL_HPP
#define XENGINE_OP_AVGPOOL_HPP

#include "pool.hpp"

class AveragePool : public Pool {
public:
  AveragePool(string n, vector<string> i, vector<string> o, memory::dims s,
              memory::dims k, memory::dims p,
              unordered_map<string, shared_ptr<Tensor>> &tensors, int training)
      : Pool(n, "AveragePool", i, o, s, k, p, tensors, training) {}
};
#endif
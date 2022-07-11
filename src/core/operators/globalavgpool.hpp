#ifndef XENGINE_OP_GLOBAL_AVGPOOL_HPP
#define XENGINE_OP_GLOBAL_AVGPOOL_HPP

#include "pool.hpp"

class GlobalAveragePool : public AveragePool {
public:
  GlobalAveragePool(string n, vector<string> i, vector<string> o,
                    memory::dims k,
                    unordered_map<string, shared_ptr<Tensor>> &tensors,
                    int training)
      : AveragePool(n, i, o, {1, 1}, k, {0, 0, 0, 0}, tensors, training) {}
};
#endif
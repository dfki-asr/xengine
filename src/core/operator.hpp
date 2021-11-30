#ifndef XENGINE_OP_HPP
#define XENGINE_OP_HPP

#include "../example_utils.hpp"
#include "common.hpp"
#include "device.hpp"
#include "tensor.hpp"

using namespace std;
using namespace dnnl;

class ExecutionOp {
public:
  ExecutionOp()
      : name("undefined"), type("undefined"), input(vector<string>()),
        output(vector<string>()) {
    memory_consumption = 0;
  }
  ExecutionOp(string n, string t, vector<string> i, vector<string> o)
      : name(n), type(t), input(i), output(o) {
    memory_consumption = 0;
  }
  string name;
  string type;
  vector<string> input;
  vector<string> output;
  float memory_consumption;
  unordered_map<string, float> timings;
  ~ExecutionOp() {}
};

class Operator {
public:
  Operator(string n, string t, vector<string> i, vector<string> o,
           unordered_map<string, unique_ptr<Tensor>> &tensors, int train)
      : name(n), type(t), input(i), output(o), training(train) {}
  string name;
  string type;
  int training;
  vector<string> input;
  vector<string> output;
  unordered_map<string, unordered_map<string, float>> timings;
  unordered_map<string, float> memory_consumption;
  ExecutionOp _f_op;
  ExecutionOp _b_op;
  string getForwardTimeName(const engine &eng) {
    return "fwd_" + getDeviceName(eng);
  }
  string getBackwardTimeName(const engine &eng) {
    return "bwd_" + getDeviceName(eng);
  }
  string getDeviceName(const engine &eng) {
    return (eng.get_kind() == dnnl::engine::kind::cpu) ? "cpu_0" : "gpu_0";
  }
  prop_kind getMode(const int training) {
    return training ? prop_kind::forward_training
                    : prop_kind::forward_inference;
  }
  memory::desc getDesc(memory::dims dims,
                       memory::format_tag tag = memory::format_tag::any) {
    return memory::desc(dims, g_data_type, tag);
  }
  void init(unordered_map<string, unique_ptr<Tensor>> &tensors) {
    // forward
    for (auto i : _f_op.input) {
      tensors[i]->add_consumer(_f_op.name);
    }
    for (auto i : _f_op.output) {
      tensors[i]->producer = _f_op.name;
      _f_op.memory_consumption +=
          product(tensors[i]->dims()) * sizeof(g_data_type);
    }
    if (training) {
      // backward
      for (auto i : _b_op.input) {
        if (tensors.find(i) != tensors.end()) {
          tensors[i]->add_consumer(_b_op.name);
        }
      }
      for (auto i : _b_op.output) {
        if (tensors.find(i) != tensors.end()) {
          tensors[i]->producer = _b_op.name;
          _b_op.memory_consumption +=
              product(tensors[i]->dims()) * sizeof(g_data_type);
        }
      }
    }
  }
  float getFwdMemoryConsumption() { return _f_op.memory_consumption; }
  float getBwdMemoryConsumption() { return _b_op.memory_consumption; }
  virtual void forward(Device &dev,
                       unordered_map<string, unique_ptr<Tensor>> &tensors,
                       memory::format_tag tag = memory::format_tag::any,
                       int measure_time = 0) = 0;
  virtual void backward(Device &dev,
                        unordered_map<string, unique_ptr<Tensor>> &tensors,
                        memory::format_tag tag = memory::format_tag::any,
                        int measure_time = 0) = 0;
  virtual void reset_fwd_primitives() {}
  virtual void reset_bwd_primitives() {}
  virtual ~Operator() = default;
};

class Operator_With_Weights : public Operator {
public:
  Operator_With_Weights(string n, string t, vector<string> i, vector<string> o,
                        unordered_map<string, unique_ptr<Tensor>> &tensors,
                        int training)
      : Operator(n, t, i, o, tensors, training) {
    auto b_i = vector<string>{"diff_" + o.at(0)};
    for (auto inp : i) {
      b_i.push_back(inp);
    }
    auto b_o = vector<string>();
    for (auto inp : i) {
      b_o.push_back("diff_" + inp);
    }
    _f_op = ExecutionOp("fwd_" + n, "fwd", i, o);
    _b_op = ExecutionOp("bwd_" + n, "bwd", b_i, b_o);
    init(tensors);
  }

  int has_bias() { return input.size() > 2; }
};
#endif
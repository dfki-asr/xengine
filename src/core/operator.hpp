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
           unordered_map<string, shared_ptr<Tensor>> &tensors, int train)
      : name(n), type(t), input(i), output(o), training(train),
        _f_device(nullptr), _b_device(nullptr), _track_only_tensor_memory(1) {}
  string name;
  string type;
  int training;
  vector<string> input;
  vector<string> output;
  unordered_map<string, unordered_map<string, float>> timings;
  unordered_map<string, float> memory_consumption;
  ExecutionOp _f_op;
  ExecutionOp _b_op;
  shared_ptr<Device> _f_device;
  shared_ptr<Device> _b_device;
  unsigned int _track_only_tensor_memory;
  string getForwardTimeName(const string dev_name) { return "fwd_" + dev_name; }
  string getBackwardTimeName(const string dev_name) {
    return "bwd_" + dev_name;
  }
  prop_kind getMode(const int training) {
    return training ? prop_kind::forward_training
                    : prop_kind::forward_inference;
  }
  memory::desc getDesc(memory::dims dims,
                       memory::format_tag tag = memory::format_tag::any) {
    return memory::desc(dims, g_data_type, tag);
  }
  void init(unordered_map<string, shared_ptr<Tensor>> &tensors) {
    // forward
    for (auto i : _f_op.input) {
      tensors[i]->add_consumer(_f_op.name);
    }
    for (auto i : _f_op.output) {
      tensors[i]->set_producer(_f_op.name);
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
          tensors[i]->set_producer(_b_op.name);
          _b_op.memory_consumption +=
              product(tensors[i]->dims()) * sizeof(g_data_type);
        }
      }
    }
  }

  long long
  getFwdMemoryConsumption(unordered_map<string, shared_ptr<Tensor>> &tensors) {
    long long memory_usage = 0;
    for (auto i : _f_op.input) {
      if (tensors[i]->producer() == "external") {
        memory_usage += tensors[i]->get_size();
      }
    }
    for (auto i : _f_op.output) {
      memory_usage += tensors[i]->get_size();
    }
    return memory_usage;
  }

  long long
  getBwdMemoryConsumption(unordered_map<string, shared_ptr<Tensor>> &tensors) {
    long long memory_usage = 0;
    for (auto i : _b_op.input) {
      if (tensors[i]->producer() == "external") {
        memory_usage += tensors[i]->get_size();
      }
    }
    for (auto i : _b_op.output) {
      memory_usage += tensors[i]->get_size();
    }
    return memory_usage;
  }

  virtual void forward(shared_ptr<Device> dev,
                       unordered_map<string, shared_ptr<Tensor>> &tensors,
                       memory::format_tag tag = memory::format_tag::any,
                       int measure_time = 0) = 0;
  virtual void backward(shared_ptr<Device> dev,
                        unordered_map<string, shared_ptr<Tensor>> &tensors,
                        memory::format_tag tag = memory::format_tag::any,
                        int measure_time = 0) = 0;
  virtual void reset_fwd_primitives() {}
  virtual void reset_bwd_primitives() {}
  virtual ~Operator() = default;
};

class Operator_With_Weights : public Operator {
public:
  Operator_With_Weights(string n, string t, vector<string> i, vector<string> o,
                        unordered_map<string, shared_ptr<Tensor>> &tensors,
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
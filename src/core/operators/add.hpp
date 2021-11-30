#ifndef XENGINE_OP_ADD_HPP
#define XENGINE_OP_ADD_HPP

#include "../operator.hpp"

class kernel_tag;

struct AddFwdContext {
  vector<shared_ptr<memory>> src_mem;
  shared_ptr<memory> dst_mem;
  shared_ptr<sum::primitive_desc> fwd_pd;
  shared_ptr<sum> add_fwd;

  AddFwdContext() : dst_mem(nullptr), fwd_pd(nullptr), add_fwd(nullptr) {}

  ~AddFwdContext() {
    for (auto i = 0; i < src_mem.size(); i++) {
      src_mem.at(i).reset();
    }
    dst_mem.reset();
    fwd_pd.reset();
    add_fwd.reset();
  }
};

class Add : public Operator {
public:
  Add(string n, vector<string> i, vector<string> o,
      unordered_map<string, unique_ptr<Tensor>> &tensors, int training)
      : Operator(n, "Add", i, o, tensors, training) {
    _f_op = ExecutionOp("fwd_" + n, "fwd", i, o);
    _b_op = ExecutionOp("bwd_" + n, "bwd", vector<string>{"diff_" + o.at(0)},
                        vector<string>{"diff_" + i.at(0), "diff_" + i.at(1)});
    _fwd_context = nullptr;
    init(tensors);
  }
  ~Add() { reset_fwd_primitives(); }
  void reset_fwd_primitives() { _fwd_context.reset(); }

  void forward(Device &dev, unordered_map<string, unique_ptr<Tensor>> &tensors,
               memory::format_tag outputTag, const int measure_time) {
    auto begin = get_time();
    auto eng = dev.get_engine();
    auto out_name = _f_op.output.at(0);
    auto time_name = getForwardTimeName(eng);
    if (_fwd_context == nullptr) {
      auto time_create = get_time();
      _fwd_context.reset(new AddFwdContext());
      timings[time_name]["create"] = get_elapsed_ms(time_create);
    }
    auto s = stream(eng);
    vector<memory::desc> src_descs;
    vector<float> scales;
    for (size_t i = 0; i < _f_op.input.size(); ++i) {
      auto src_name = _f_op.input.at(i);
      auto src_desc = tensors[src_name]->desc();
      src_descs.push_back(src_desc);
      scales.push_back(1.0f);
      if (i >= _fwd_context->src_mem.size()) {
        _fwd_context->src_mem.push_back(shared_ptr<memory>(nullptr));
      }
      if (_fwd_context->src_mem.at(i) == nullptr) {
        auto time_create = get_time();
        _fwd_context->src_mem.at(i).reset(new memory(src_desc, eng));
        timings[time_name]["create"] += get_elapsed_ms(time_create);
      }
    }
    if (_fwd_context->fwd_pd == nullptr) {
      auto time_create = get_time();
      _fwd_context->fwd_pd.reset(
          new sum::primitive_desc(scales, src_descs, eng));
      _fwd_context->add_fwd.reset(new sum(*_fwd_context->fwd_pd));
      _fwd_context->dst_mem.reset(
          new memory(_fwd_context->fwd_pd.get()->dst_desc(), eng));
      tensors[out_name]->init(_fwd_context->fwd_pd.get()->dst_desc(), eng);
      timings[time_name]["create"] += get_elapsed_ms(time_create);
    }
    // reorders
    unordered_map<int, memory> args;
    for (size_t i = 0; i < _f_op.input.size(); ++i) {
      auto src_name = _f_op.input.at(i);
      timings[time_name][src_name] =
          maybe_do_reorder(tensors[src_name]->get_memory(),
                           *_fwd_context->src_mem.at(i), s, measure_time);
      args[DNNL_ARG_MULTIPLE_SRC + i] = *_fwd_context->src_mem.at(i);
    }
    args[DNNL_ARG_DST] = *_fwd_context->dst_mem;
    // execute
    auto time_exe = get_time();
    _fwd_context->add_fwd->execute(s, args);
    s.wait();
    if (measure_time) {
      timings[time_name]["exe"] = get_elapsed_ms(time_exe);
      timings[time_name]["total"] = get_elapsed_ms(begin);
    }
  }

  void backward(Device &dev, unordered_map<string, unique_ptr<Tensor>> &tensors,
                memory::format_tag outputTag, const int measure_time) {
    auto begin = get_time();
    auto eng = dev.get_engine();
    auto out_diff_a_name = _b_op.output.at(0);
    auto out_diff_b_name = _b_op.output.at(1);
    auto in_diff_name = _b_op.input.at(0);
    auto src_a_md = tensors[_f_op.input.at(0)]->desc();
    auto src_b_md = tensors[_f_op.input.at(1)]->desc();
    assert(src_a_md == src_b_md);
    assert(tensors.find(in_diff_name) != tensors.end());
    auto time_name = getBackwardTimeName(eng);
    // get memory
    auto in_diff_mem = make_memory(src_a_md, eng);
    // reorders
    auto s = stream(eng);
    timings[time_name][in_diff_name] = maybe_do_reorder(
        tensors[in_diff_name]->get_memory(), in_diff_mem, s, measure_time);
    // execute
    tensors[out_diff_a_name]->init(src_a_md, eng);
    tensors[out_diff_b_name]->init(src_b_md, eng);
    tensors[out_diff_a_name]->set_memory(in_diff_mem);
    tensors[out_diff_b_name]->set_memory(in_diff_mem);
    timings[time_name]["create"] = 0.0f;
    if (measure_time) {
      timings[time_name]["exe"] = get_elapsed_ms(begin);
      timings[time_name]["total"] = get_elapsed_ms(begin);
    }
  }
  shared_ptr<AddFwdContext> _fwd_context;
};
#endif
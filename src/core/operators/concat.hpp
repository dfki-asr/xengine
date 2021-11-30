#ifndef XENGINE_OP_CONCAT_HPP
#define XENGINE_OP_CONCAT_HPP

#include "../operator.hpp"

struct ConcatFwdContext {
  vector<shared_ptr<memory>> src_mem;
  shared_ptr<memory> dst_mem;
  shared_ptr<concat::primitive_desc> fwd_pd;
  shared_ptr<concat> concat_fwd;

  ConcatFwdContext() : dst_mem(nullptr), fwd_pd(nullptr), concat_fwd(nullptr) {}

  ~ConcatFwdContext() {
    for (auto i = 0; i < src_mem.size(); i++) {
      src_mem.at(i).reset();
    }
    dst_mem.reset();
    fwd_pd.reset();
    concat_fwd.reset();
  }
};

class Concat : public Operator {
public:
  Concat(string n, vector<string> i, vector<string> o, int a,
         unordered_map<string, unique_ptr<Tensor>> &tensors, int training)
      : Operator(n, "Concat", i, o, tensors, training) {
    axis = a;
    _f_op = ExecutionOp("fwd_" + n, "fwd", i, o);
    _fwd_context = nullptr;
    init(tensors);
  }
  ~Concat() { reset_fwd_primitives(); }
  void reset_fwd_primitives() { _fwd_context.reset(); }

  void forward(Device &dev, unordered_map<string, unique_ptr<Tensor>> &tensors,
               memory::format_tag outputTag, const int measure_time) {
    auto begin = get_time();
    auto eng = dev.get_engine();
    auto out_name = _f_op.output.at(0);
    auto time_name = getForwardTimeName(eng);
    if (_fwd_context == nullptr) {
      auto time_create = get_time();
      _fwd_context.reset(new ConcatFwdContext());
      timings[time_name]["create"] = get_elapsed_ms(time_create);
    }
    auto s = stream(eng);
    vector<memory::desc> src_descs;
    for (size_t i = 0; i < _f_op.input.size(); ++i) {
      auto src_name = _f_op.input.at(i);
      auto src_desc = tensors[src_name]->desc();
      src_descs.push_back(src_desc);
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
          new concat::primitive_desc(axis, src_descs, eng));
      _fwd_context->concat_fwd.reset(new concat(*_fwd_context->fwd_pd));
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
    _fwd_context->concat_fwd->execute(s, args);
    s.wait();
    if (measure_time) {
      timings[time_name]["exe"] = get_elapsed_ms(time_exe);
      timings[time_name]["total"] = get_elapsed_ms(begin);
    }
  }

  void backward(Device &dev, unordered_map<string, unique_ptr<Tensor>> &tensors,
                memory::format_tag outputTag, const int measure_time) {}
  int axis;
  shared_ptr<ConcatFwdContext> _fwd_context;
};
#endif
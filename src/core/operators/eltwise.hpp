#ifndef XENGINE_OP_ELTWISE_HPP
#define XENGINE_OP_ELTWISE_HPP

#include "../operator.hpp"

using namespace dnnl;

struct EltwiseFwdContext {
  shared_ptr<memory> src_mem;
  shared_ptr<memory> dst_mem;
  shared_ptr<eltwise_forward::desc> fwd_desc;
  shared_ptr<eltwise_forward::primitive_desc> fwd_pd;
  shared_ptr<eltwise_forward> relu_fwd;

  EltwiseFwdContext()
      : src_mem(nullptr), dst_mem(nullptr), fwd_desc(nullptr), fwd_pd(nullptr),
        relu_fwd(nullptr) {}

  ~EltwiseFwdContext() {
    src_mem.reset();
    dst_mem.reset();
    fwd_desc.reset();
    fwd_pd.reset();
    relu_fwd.reset();
  }
};

struct EltwiseBwdContext {
  shared_ptr<memory> in_diff_mem;
  shared_ptr<memory> src_mem;
  shared_ptr<memory> out_diff_mem;
  shared_ptr<eltwise_backward::desc> bwd_desc;
  shared_ptr<eltwise_backward::primitive_desc> bwd_pd;
  shared_ptr<eltwise_backward> relu_bwd;

  EltwiseBwdContext()
      : in_diff_mem(nullptr), src_mem(nullptr), out_diff_mem(nullptr),
        bwd_desc(nullptr), bwd_pd(nullptr), relu_bwd(nullptr) {}

  ~EltwiseBwdContext() {
    in_diff_mem.reset();
    src_mem.reset();
    in_diff_mem.reset();
    bwd_desc.reset();
    bwd_pd.reset();
    relu_bwd.reset();
  }
};

class Eltwise : public Operator {
public:
  Eltwise(string n, string t, vector<string> i, vector<string> o, float a,
          unordered_map<string, unique_ptr<Tensor>> &tensors, int training)
      : Operator(n, t, i, o, tensors, training) {
    if (t == "Relu") {
      algo = algorithm::eltwise_relu;
      alpha = a;
    } else if (t == "LeakyRelu") {
      algo = algorithm::eltwise_relu;
      alpha = a;
    } else {
      throw runtime_error("Unsupported Eltwise Type!");
    }
    _f_op = ExecutionOp("fwd_" + n, "fwd", i, o);
    _b_op = ExecutionOp("bwd_" + n, "bwd",
                        vector<string>{"diff_" + o.at(0), i.at(0)},
                        vector<string>{"diff_" + i.at(0)});
    _fwd_context = nullptr;
    _bwd_context = nullptr;
    init(tensors);
  }
  ~Eltwise() {
    reset_fwd_primitives();
    reset_bwd_primitives();
  }
  void reset_fwd_primitives() { _fwd_context.reset(); }
  void reset_bwd_primitives() { _bwd_context.reset(); }

  void forward(Device &dev, unordered_map<string, unique_ptr<Tensor>> &tensors,
               memory::format_tag outputTag, const int measure_time) {
    auto begin = get_time();
    auto eng = dev.get_engine();
    auto src_name = _f_op.input.at(0);
    auto out_name = _f_op.output.at(0);
    auto src_md = tensors[src_name]->desc();
    auto time_name = getForwardTimeName(eng);
    if (_fwd_context == nullptr) {
      auto time_create = get_time();
      _fwd_context.reset(new EltwiseFwdContext());
      _fwd_context->fwd_desc.reset(
          new eltwise_forward::desc({getMode(training), algo, src_md, alpha}));
      _fwd_context->fwd_pd.reset(
          new eltwise_forward::primitive_desc(*_fwd_context->fwd_desc, eng));
      _fwd_context->relu_fwd.reset(new eltwise_forward(*_fwd_context->fwd_pd));
      // get memory
      _fwd_context->src_mem.reset(
          new memory(_fwd_context->fwd_pd.get()->src_desc(), eng));
      _fwd_context->dst_mem.reset(
          new memory(_fwd_context->fwd_pd.get()->dst_desc(), eng));
      tensors[out_name]->init(_fwd_context->fwd_pd.get()->dst_desc(), eng);
      timings[time_name]["create"] = get_elapsed_ms(time_create);
    }
    // reorders
    auto s = stream(eng);
    timings[time_name][src_name] =
        maybe_do_reorder(tensors[src_name]->get_memory(),
                         *_fwd_context->src_mem, s, measure_time);
    // execute
    auto args =
        unordered_map<int, memory>({{DNNL_ARG_SRC, *_fwd_context->src_mem},
                                    {DNNL_ARG_DST, *_fwd_context->dst_mem}});
    auto time_exe = get_time();
    _fwd_context->relu_fwd->execute(s, args);
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
    auto src_name = _b_op.input.at(1);
    auto out_name = _f_op.output.at(0);
    auto in_diff_name = _b_op.input.at(0);
    auto out_diff_name = _b_op.output.at(0);
    auto src_md = tensors[src_name]->desc();
    auto dst_md = tensors[out_name]->desc();
    auto time_name = getBackwardTimeName(eng);
    if (_bwd_context == nullptr) {
      auto time_create = get_time();
      _bwd_context.reset(new EltwiseBwdContext());
      _bwd_context->bwd_desc.reset(
          new eltwise_backward::desc({algo, src_md, dst_md, alpha}));
      int reuse_fwd_pd = 0;
      if (_fwd_context != nullptr) {
        reuse_fwd_pd = _fwd_context->fwd_pd->get_engine() == eng ? 1 : 0;
      }
      auto fwd_pd =
          reuse_fwd_pd
              ? *_fwd_context->fwd_pd
              : eltwise_forward::primitive_desc(
                    {prop_kind::forward_training, algo, src_md, alpha}, eng);
      _bwd_context->bwd_pd.reset(new eltwise_backward::primitive_desc(
          *_bwd_context->bwd_desc, eng, fwd_pd));
      _bwd_context->relu_bwd.reset(new eltwise_backward(*_bwd_context->bwd_pd));
      // get memory
      _bwd_context->src_mem.reset(
          new memory(_bwd_context->bwd_pd.get()->src_desc(), eng));
      _bwd_context->in_diff_mem.reset(
          new memory(_bwd_context->bwd_pd.get()->src_desc(), eng));
      _bwd_context->out_diff_mem.reset(
          new memory(_bwd_context->bwd_pd.get()->src_desc(), eng));
      tensors[out_diff_name]->init(_bwd_context->bwd_pd.get()->src_desc(), eng);
      timings[time_name]["create"] = get_elapsed_ms(time_create);
    }
    // reorders
    auto s = stream(eng);
    timings[time_name][src_name] =
        maybe_do_reorder(tensors[src_name]->get_memory(),
                         *_bwd_context->src_mem, s, measure_time);
    timings[time_name][in_diff_name] =
        maybe_do_reorder(tensors[in_diff_name]->get_memory(),
                         *_bwd_context->in_diff_mem, s, measure_time);
    // execute
    auto args = unordered_map<int, memory>(
        {{DNNL_ARG_SRC, *_bwd_context->src_mem},
         {DNNL_ARG_DIFF_DST, *_bwd_context->in_diff_mem},
         {DNNL_ARG_DIFF_SRC, *_bwd_context->out_diff_mem}});
    auto time_exe = get_time();
    _bwd_context->relu_bwd->execute(s, args);
    s.wait();
    if (measure_time) {
      timings[time_name]["exe"] = get_elapsed_ms(time_exe);
      timings[time_name]["total"] = get_elapsed_ms(begin);
    }
  }
  algorithm algo;
  float alpha;
  shared_ptr<EltwiseFwdContext> _fwd_context;
  shared_ptr<EltwiseBwdContext> _bwd_context;
};
#endif
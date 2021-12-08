#ifndef XENGINE_OP_LRN_HPP
#define XENGINE_OP_LRN_HPP

#include "../operator.hpp"

using namespace dnnl;

struct LRNFwdContext {
  shared_ptr<memory> src_mem;
  shared_ptr<memory> dst_mem;
  shared_ptr<memory> ws_mem;
  shared_ptr<lrn_forward::desc> fwd_desc;
  shared_ptr<lrn_forward::primitive_desc> fwd_pd;
  shared_ptr<lrn_forward> lrn_fwd;

  LRNFwdContext()
      : src_mem(nullptr), dst_mem(nullptr), ws_mem(nullptr), fwd_desc(nullptr),
        fwd_pd(nullptr), lrn_fwd(nullptr) {}

  ~LRNFwdContext() {
    src_mem.reset();
    dst_mem.reset();
    ws_mem.reset();
    fwd_desc.reset();
    fwd_pd.reset();
    lrn_fwd.reset();
  }
};

struct LRNBwdContext {
  shared_ptr<memory> in_diff_mem;
  shared_ptr<memory> src_mem;
  shared_ptr<memory> ws_mem;
  shared_ptr<memory> out_diff_mem;
  shared_ptr<lrn_backward::desc> bwd_desc;
  shared_ptr<lrn_backward::primitive_desc> bwd_pd;
  shared_ptr<lrn_backward> lrn_bwd;

  LRNBwdContext()
      : in_diff_mem(nullptr), src_mem(nullptr), ws_mem(nullptr),
        out_diff_mem(nullptr), bwd_desc(nullptr), bwd_pd(nullptr),
        lrn_bwd(nullptr) {}

  ~LRNBwdContext() {
    in_diff_mem.reset();
    src_mem.reset();
    ws_mem.reset();
    in_diff_mem.reset();
    bwd_desc.reset();
    bwd_pd.reset();
    lrn_bwd.reset();
  }
};

class LRN : public Operator {
public:
  LRN(string n, vector<string> i, vector<string> o, float a, float be, float bi,
      int s, unordered_map<string, shared_ptr<Tensor>> &tensors, int training)
      : Operator(n, "LRN", i, o, tensors, training) {
    alpha = a;
    beta = be;
    bias = bi;
    size = s;
    algo = algorithm::lrn_across_channels;
    auto f_o = vector<string>{o.at(0)};
    if (training) {
      f_o.push_back(o.at(0) + "_ws");
    }
    _f_op = ExecutionOp("fwd_" + n, "fwd", i, f_o);
    _b_op =
        ExecutionOp("bwd_" + n, "bwd",
                    vector<string>{"diff_" + o.at(0), i.at(0), o.at(0) + "_ws"},
                    vector<string>{"diff_" + i.at(0)});
    _fwd_context = nullptr;
    _bwd_context = nullptr;
    init(tensors);
    if (training) {
      auto ws_name = _f_op.output.at(1);
      tensors[ws_name]->add_consumer(_b_op.name);
    }
  }
  ~LRN() {
    reset_fwd_primitives();
    reset_bwd_primitives();
  }
  void reset_fwd_primitives() { _fwd_context.reset(); }
  void reset_bwd_primitives() { _bwd_context.reset(); }

  void forward(Device &dev, unordered_map<string, shared_ptr<Tensor>> &tensors,
               memory::format_tag outputTag, const int measure_time) {
    auto begin = get_time();
    auto eng = dev.get_engine();
    auto src_name = _f_op.input.at(0);
    auto out_name = _f_op.output.at(0);
    auto src_md = tensors[src_name]->desc();
    auto time_name = getForwardTimeName(eng);
    if (_fwd_context == nullptr) {
      auto time_create = get_time();
      // Create operation descriptor.
      _fwd_context.reset(new LRNFwdContext());
      _fwd_context->fwd_desc.reset(new lrn_forward::desc(
          {getMode(training), algo, src_md, size, alpha, beta, bias}));
      _fwd_context->fwd_pd.reset(
          new lrn_forward::primitive_desc(*_fwd_context->fwd_desc, eng));
      _fwd_context->lrn_fwd.reset(new lrn_forward(*_fwd_context->fwd_pd));
      // get memory
      _fwd_context->src_mem.reset(
          new memory(_fwd_context->fwd_pd.get()->src_desc(), eng));
      _fwd_context->dst_mem.reset(
          new memory(_fwd_context->fwd_pd.get()->dst_desc(), eng));
      tensors[out_name]->init(_fwd_context->fwd_pd.get()->dst_desc(), eng);
      if (training) {
        auto ws_name = _f_op.output.at(1);
        if (_fwd_context->ws_mem == nullptr) {
          _fwd_context->ws_mem.reset(
              new memory(_fwd_context->fwd_pd.get()->workspace_desc(), eng));
          tensors[ws_name]->init(_fwd_context->fwd_pd.get()->workspace_desc(),
                                 eng);
        }
      }
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
    if (training) {
      auto ws_name = _f_op.output.at(1);
      timings[time_name][ws_name] =
          maybe_do_reorder(tensors[ws_name]->get_memory(),
                           *_fwd_context->ws_mem, s, measure_time);
      args.insert({DNNL_ARG_WORKSPACE, *_fwd_context->ws_mem});
    }

    auto time_exe = get_time();
    _fwd_context->lrn_fwd->execute(s, args);
    s.wait();
    if (measure_time) {
      timings[time_name]["exe"] = get_elapsed_ms(time_exe);
      timings[time_name]["total"] = get_elapsed_ms(begin);
    }
  }

  void backward(Device &dev, unordered_map<string, shared_ptr<Tensor>> &tensors,
                memory::format_tag outputTag, const int measure_time) {
    auto begin = get_time();
    auto eng = dev.get_engine();
    auto src_name = _b_op.input.at(1);
    auto ws_name = _b_op.input.at(2);
    auto out_name = _f_op.output.at(0);
    auto in_diff_name = _b_op.input.at(0);
    auto out_diff_name = _b_op.output.at(0);
    auto src_md = tensors[src_name]->desc();
    auto dst_md = tensors[out_name]->desc();
    auto time_name = getBackwardTimeName(eng);
    if (_bwd_context == nullptr) {
      auto time_create = get_time();
      _bwd_context.reset(new LRNBwdContext());
      _bwd_context->bwd_desc.reset(new lrn_backward::desc(
          {algo, src_md, dst_md, size, alpha, beta, bias}));
      int reuse_fwd_pd = 0;
      if (_fwd_context != nullptr) {
        reuse_fwd_pd = _fwd_context->fwd_pd->get_engine() == eng ? 1 : 0;
      }
      auto fwd_pd =
          reuse_fwd_pd
              ? *_fwd_context->fwd_pd
              : lrn_forward::primitive_desc({prop_kind::forward_training, algo,
                                             src_md, size, alpha, beta, bias},
                                            eng);
      _bwd_context->bwd_pd.reset(new lrn_backward::primitive_desc(
          *_bwd_context->bwd_desc, eng, fwd_pd));
      _bwd_context->lrn_bwd.reset(new lrn_backward(*_bwd_context->bwd_pd));
      // get memory
      _bwd_context->src_mem.reset(
          new memory(_bwd_context->bwd_pd.get()->src_desc(), eng));
      _bwd_context->in_diff_mem.reset(
          new memory(_bwd_context->bwd_pd.get()->src_desc(), eng));
      _bwd_context->ws_mem.reset(new memory(tensors[ws_name]->desc(), eng));
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
    timings[time_name][ws_name] = maybe_do_reorder(
        tensors[ws_name]->get_memory(), *_bwd_context->ws_mem, s, measure_time);
    // execute
    auto args = unordered_map<int, memory>(
        {{DNNL_ARG_SRC, *_bwd_context->src_mem},
         {DNNL_ARG_DIFF_DST, *_bwd_context->in_diff_mem},
         {DNNL_ARG_DIFF_SRC, *_bwd_context->out_diff_mem},
         {DNNL_ARG_WORKSPACE, *_bwd_context->ws_mem}});
    auto time_exe = get_time();
    _bwd_context->lrn_bwd->execute(s, args);
    s.wait();
    if (measure_time) {
      timings[time_name]["exe"] = get_elapsed_ms(time_exe);
      timings[time_name]["total"] = get_elapsed_ms(begin);
    }
  }
  algorithm algo;
  float alpha;
  float beta;
  float bias;
  int size;
  shared_ptr<LRNFwdContext> _fwd_context;
  shared_ptr<LRNBwdContext> _bwd_context;
};
#endif
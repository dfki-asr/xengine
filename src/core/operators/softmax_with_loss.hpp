#ifndef XENGINE_OP_SOFTMAXWITHLOSS_HPP
#define XENGINE_OP_SOFTMAXWITHLOSS_HPP

#include "../operator.hpp"

struct SoftmaxWithLossFwdContext {
  shared_ptr<memory> src_mem;
  shared_ptr<memory> labels_mem;
  shared_ptr<memory> dst_mem;
  shared_ptr<memory> loss_mem;
  shared_ptr<softmax_forward::desc> fwd_desc;
  shared_ptr<softmax_forward::primitive_desc> fwd_pd;
  shared_ptr<softmax_forward> softmax_fwd;

  SoftmaxWithLossFwdContext()
      : src_mem(nullptr), labels_mem(nullptr), dst_mem(nullptr),
        loss_mem(nullptr), fwd_desc(nullptr), fwd_pd(nullptr),
        softmax_fwd(nullptr) {}

  ~SoftmaxWithLossFwdContext() {
    src_mem.reset();
    labels_mem.reset();
    dst_mem.reset();
    loss_mem.reset();
    fwd_desc.reset();
    fwd_pd.reset();
    softmax_fwd.reset();
  }
};

struct SoftmaxWithLossBwdContext {
  shared_ptr<memory> in_diff_mem;
  shared_ptr<memory> src_mem;
  shared_ptr<memory> labels_mem;
  shared_ptr<memory> out_diff_mem;
  shared_ptr<softmax_backward::desc> bwd_desc;
  shared_ptr<softmax_backward::primitive_desc> bwd_pd;
  shared_ptr<softmax_backward> softmax_bwd;

  SoftmaxWithLossBwdContext()
      : in_diff_mem(nullptr), src_mem(nullptr), labels_mem(nullptr),
        out_diff_mem(nullptr), bwd_desc(nullptr), bwd_pd(nullptr),
        softmax_bwd(nullptr) {}

  ~SoftmaxWithLossBwdContext() {
    in_diff_mem.reset();
    src_mem.reset();
    labels_mem.reset();
    out_diff_mem.reset();
    bwd_desc.reset();
    bwd_pd.reset();
    softmax_bwd.reset();
  }
};

class SoftmaxWithLoss : public Operator {
public:
  SoftmaxWithLoss(string n, vector<string> i, vector<string> o, int a,
                  unordered_map<string, shared_ptr<Tensor>> &tensors,
                  int training)
      : Operator(n, "SoftmaxWithLoss", i, o, tensors, training) {
    axis = a;
    _f_op = ExecutionOp("fwd_" + n, "fwd", vector<string>{i.at(0), i.at(1)},
                        vector<string>{o.at(0), o.at(1)});
    _b_op = ExecutionOp("bwd_" + n, "bwd",
                        vector<string>{o.at(1), i.at(0), i.at(1)},
                        vector<string>{"diff_" + i.at(0)});
    _fwd_context = nullptr;
    _bwd_context = nullptr;
    init(tensors);
    tensors[_f_op.output.at(0)]->add_consumer("external");
  }
  ~SoftmaxWithLoss() {
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
    auto labels_name = _f_op.input.at(1);
    auto loss_name = _f_op.output.at(1);
    auto src_md = tensors[src_name]->desc();
    auto time_name = getForwardTimeName(eng);
    if (_fwd_context == nullptr) {
      auto time_create = get_time();
      // cout << "Create context SoftmaxWithLoss fwd" << endl;
      _fwd_context.reset(new SoftmaxWithLossFwdContext());
      _fwd_context->fwd_desc.reset(
          new softmax_forward::desc(getMode(training), src_md, axis));
      _fwd_context->fwd_pd.reset(
          new softmax_forward::primitive_desc(*_fwd_context->fwd_desc, eng));
      _fwd_context->softmax_fwd.reset(
          new softmax_forward(*_fwd_context->fwd_pd));
      // get memory
      _fwd_context->src_mem.reset(
          new memory(_fwd_context->fwd_pd.get()->src_desc(), eng));
      _fwd_context->labels_mem.reset(
          new memory(tensors[labels_name]->desc(), eng));
      _fwd_context->dst_mem.reset(
          new memory(_fwd_context->fwd_pd.get()->dst_desc(), eng));
      tensors[out_name]->init(_fwd_context->fwd_pd.get()->dst_desc(), eng);
      _fwd_context->loss_mem.reset(
          new memory(tensors[labels_name]->desc(), eng));
      tensors[loss_name]->init(tensors[labels_name]->desc(), eng);
      timings[time_name]["create"] = get_elapsed_ms(time_create);
    }
    // reorders
    auto s = stream(eng);
    timings[time_name][src_name] =
        maybe_do_reorder(tensors[src_name]->get_memory(),
                         *_fwd_context->src_mem, s, measure_time);
    timings[time_name][labels_name] =
        maybe_do_reorder(tensors[labels_name]->get_memory(),
                         *_fwd_context->labels_mem, s, measure_time);
    // execute
    auto args =
        unordered_map<int, memory>({{DNNL_ARG_SRC, *_fwd_context->src_mem},
                                    {DNNL_ARG_DST, *_fwd_context->dst_mem}});
    auto time_exe = get_time();
    _fwd_context->softmax_fwd->execute(s, args);
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
    auto out_name = _f_op.output.at(0);
    auto labels_name = _b_op.input.at(2);
    auto in_diff_name = _b_op.input.at(0);
    auto out_diff_name = _b_op.output.at(0);
    auto src_md = tensors[src_name]->desc();
    auto dst_md = tensors[out_name]->desc();
    auto l_md = tensors[labels_name]->desc();
    auto time_name = getBackwardTimeName(eng);
    if (_bwd_context == nullptr) {
      auto time_create = get_time();
      _bwd_context.reset(new SoftmaxWithLossBwdContext());
      _bwd_context->bwd_desc.reset(
          new softmax_backward::desc({src_md, dst_md, axis}));
      int reuse_fwd_pd = 0;
      if (_fwd_context != nullptr) {
        reuse_fwd_pd = _fwd_context->fwd_pd->get_engine() == eng ? 1 : 0;
      }
      auto fwd_pd = reuse_fwd_pd
                        ? *_fwd_context->fwd_pd
                        : softmax_forward::primitive_desc(
                              {prop_kind::forward_training, src_md, axis}, eng);
      _bwd_context->bwd_pd.reset(new softmax_backward::primitive_desc(
          *_bwd_context->bwd_desc, eng, fwd_pd));
      _bwd_context->softmax_bwd.reset(
          new softmax_backward(*_bwd_context->bwd_pd));
      // get memory
      _bwd_context->in_diff_mem.reset(new memory(l_md, eng));
      _bwd_context->src_mem.reset(
          new memory(_bwd_context->bwd_pd.get()->dst_desc(), eng));
      _bwd_context->labels_mem.reset(new memory(l_md, eng));
      _bwd_context->out_diff_mem.reset(new memory(src_md, eng));
      tensors[out_diff_name]->init(src_md, eng);
      timings[time_name]["create"] = get_elapsed_ms(time_create);
    }
    // reorders
    auto s = stream(eng);
    timings[time_name][src_name] =
        maybe_do_reorder(tensors[src_name]->get_memory(),
                         *_bwd_context->src_mem, s, measure_time);
    timings[time_name][labels_name] =
        maybe_do_reorder(tensors[labels_name]->get_memory(),
                         *_bwd_context->labels_mem, s, measure_time);
    timings[time_name][in_diff_name] =
        maybe_do_reorder(tensors[in_diff_name]->get_memory(),
                         *_bwd_context->in_diff_mem, s, measure_time);
    // execute
    auto args = unordered_map<int, memory>(
        {{DNNL_ARG_SRC, *_bwd_context->src_mem},
         {DNNL_ARG_DST, *_bwd_context->labels_mem},
         {DNNL_ARG_DIFF_DST, *_bwd_context->in_diff_mem},
         {DNNL_ARG_DIFF_SRC, *_bwd_context->out_diff_mem}});
    auto time_exe = get_time();
    _bwd_context->softmax_bwd->execute(s, args);
    s.wait();
    if (measure_time) {
      timings[time_name]["exe"] = get_elapsed_ms(time_exe);
      timings[time_name]["total"] = get_elapsed_ms(begin);
    }
  }
  int axis;
  shared_ptr<SoftmaxWithLossFwdContext> _fwd_context;
  shared_ptr<SoftmaxWithLossBwdContext> _bwd_context;
};
#endif
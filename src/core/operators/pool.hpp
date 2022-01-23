#ifndef XENGINE_OP_POOL_HPP
#define XENGINE_OP_POOL_HPP

#include "../operator.hpp"

struct PoolFwdContext {
  shared_ptr<memory> src_mem;
  shared_ptr<memory> dst_mem;
  shared_ptr<memory> ws_mem;
  shared_ptr<pooling_forward::desc> fwd_desc;
  shared_ptr<pooling_forward::primitive_desc> fwd_pd;
  shared_ptr<pooling_forward> pool_fwd;

  PoolFwdContext()
      : src_mem(nullptr), dst_mem(nullptr), ws_mem(nullptr), fwd_desc(nullptr),
        fwd_pd(nullptr), pool_fwd(nullptr) {}

  ~PoolFwdContext() {
    src_mem.reset();
    dst_mem.reset();
    ws_mem.reset();
    fwd_desc.reset();
    fwd_pd.reset();
    pool_fwd.reset();
  }

  size_t get_memory_used() {
    size_t memory_used_bytes = 0;
    if (src_mem != nullptr)
      memory_used_bytes += src_mem->get_desc().get_size();
    if (dst_mem != nullptr)
      memory_used_bytes += dst_mem->get_desc().get_size();
    if (ws_mem != nullptr)
      memory_used_bytes += ws_mem->get_desc().get_size();
    return memory_used_bytes;
  }
};

struct PoolBwdContext {
  shared_ptr<memory> in_diff_mem;
  shared_ptr<memory> ws_mem;
  shared_ptr<memory> out_diff_mem;
  shared_ptr<pooling_backward::desc> bwd_desc;
  shared_ptr<pooling_backward::primitive_desc> bwd_pd;
  shared_ptr<pooling_backward> pool_bwd;

  PoolBwdContext()
      : in_diff_mem(nullptr), ws_mem(nullptr), out_diff_mem(nullptr),
        bwd_desc(nullptr), bwd_pd(nullptr), pool_bwd(nullptr) {}

  ~PoolBwdContext() {
    in_diff_mem.reset();
    ws_mem.reset();
    out_diff_mem.reset();
    bwd_desc.reset();
    bwd_pd.reset();
    pool_bwd.reset();
  }

  size_t get_memory_used() {
    size_t memory_used_bytes = 0;
    if (in_diff_mem != nullptr)
      memory_used_bytes += in_diff_mem->get_desc().get_size();
    if (ws_mem != nullptr)
      memory_used_bytes += ws_mem->get_desc().get_size();
    if (out_diff_mem != nullptr)
      memory_used_bytes += out_diff_mem->get_desc().get_size();
    return memory_used_bytes;
  }
};

class Pool : public Operator {
public:
  Pool(string n, string t, vector<string> i, vector<string> o, memory::dims s,
       memory::dims k, memory::dims p,
       unordered_map<string, shared_ptr<Tensor>> &tensors, int training)
      : Operator(n, t, i, o, tensors, training) {
    stride = s;
    kernel = k;
    padding_l = {p.begin(), p.begin() + k.size()};
    padding_r = {p.begin() + k.size(), p.end()};
    if (type == "MaxPool") {
      algo = algorithm::pooling_max;
    } else if (type == "AveragePool") {
      algo = algorithm::pooling_avg;
    } else {
      throw runtime_error("Unsupported Pool type!");
    }
    auto f_o = vector<string>{o.at(0)};
    if (training) {
      f_o.push_back(o.at(0) + "_ws");
    }
    _f_op = ExecutionOp("fwd_" + n, "fwd", i, f_o);
    _b_op = ExecutionOp("bwd_" + n, "bwd",
                        vector<string>{"diff_" + o.at(0), o.at(0) + "_ws"},
                        vector<string>{"diff_" + i.at(0)});
    _fwd_context = nullptr;
    _bwd_context = nullptr;
    init(tensors);
    if (training) {
      auto ws_name = _f_op.output.at(1);
      tensors[ws_name]->add_consumer(_b_op.name);
    }
  }
  ~Pool() {
    reset_fwd_primitives();
    reset_bwd_primitives();
  }
  void reset_fwd_primitives() { _fwd_context.reset(); }
  void reset_bwd_primitives() { _bwd_context.reset(); }

  void forward(shared_ptr<Device> dev,
               unordered_map<string, shared_ptr<Tensor>> &tensors,
               memory::format_tag outputTag, const int measure_time) {
    auto begin = get_time();
    auto eng = dev->get_engine();
    auto src_name = _f_op.input.at(0);
    auto out_name = _f_op.output.at(0);
    auto src_dims = tensors[src_name]->dims();
    auto src_md = tensors[src_name]->desc();
    auto dst_dims = get_output_dims(src_dims, src_dims.at(1), kernel, stride,
                                    padding_l, padding_r);
    auto time_name = getForwardTimeName(eng);
    if (_fwd_context == nullptr) {
      auto time_create = get_time();
      _fwd_context.reset(new PoolFwdContext());
      _fwd_context->fwd_desc.reset(new pooling_forward::desc(
          {getMode(training), algo, src_md, getDesc(dst_dims, outputTag),
           stride, kernel, padding_l, padding_r}));
      _fwd_context->fwd_pd.reset(
          new pooling_forward::primitive_desc(*_fwd_context->fwd_desc, eng));
      _fwd_context->pool_fwd.reset(new pooling_forward(*_fwd_context->fwd_pd));
      // get memory
      _fwd_context->src_mem.reset(
          new memory(_fwd_context->fwd_pd.get()->src_desc(), eng));
      _fwd_context->dst_mem.reset(
          new memory(_fwd_context->fwd_pd.get()->dst_desc(), eng));
      tensors[out_name]->init(_fwd_context->fwd_pd.get()->dst_desc(), dev);
      if (training) {
        auto ws_name = _f_op.output.at(1);
        if (_fwd_context->ws_mem == nullptr) {
          _fwd_context->ws_mem.reset(
              new memory(_fwd_context->fwd_pd.get()->dst_desc(), eng));
          tensors[ws_name]->init(_fwd_context->fwd_pd.get()->dst_desc(), dev);
        }
      }
      timings[time_name]["create"] = get_elapsed_ms(time_create);
      dev->memory_used += _fwd_context->get_memory_used();
    }
    // reorders
    auto s = dev->get_stream(0);
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
    _fwd_context->pool_fwd->execute(s, args);
    s.wait();
    if (measure_time) {
      timings[time_name]["exe"] = get_elapsed_ms(time_exe);
      timings[time_name]["total"] = get_elapsed_ms(begin);
    }
  }

  void backward(shared_ptr<Device> dev,
                unordered_map<string, shared_ptr<Tensor>> &tensors,
                memory::format_tag outputTag, const int measure_time) {
    auto begin = get_time();
    auto eng = dev->get_engine();
    auto out_name = _f_op.output.at(0);
    auto ws_name = _b_op.input.at(1);
    auto in_diff_name = _b_op.input.at(0);
    auto out_diff_name = _b_op.output.at(0);
    auto src_md = tensors[_f_op.input.at(0)]->desc();
    auto dst_md = tensors[out_name]->desc();
    auto time_name = getBackwardTimeName(eng);
    if (_bwd_context == nullptr) {
      auto time_create = get_time();
      _bwd_context.reset(new PoolBwdContext());
      _bwd_context->bwd_desc.reset(new pooling_backward::desc(
          {algo, src_md, dst_md, stride, kernel, padding_l, padding_r}));
      int reuse_fwd_pd = 0;
      if (_fwd_context != nullptr) {
        reuse_fwd_pd = _fwd_context->fwd_pd->get_engine() == eng ? 1 : 0;
      }
      auto fwd_pd = reuse_fwd_pd
                        ? *_fwd_context->fwd_pd
                        : pooling_forward::primitive_desc(
                              {prop_kind::forward_training, algo, src_md,
                               dst_md, stride, kernel, padding_l, padding_r},
                              eng);
      _bwd_context->bwd_pd.reset(new pooling_backward::primitive_desc(
          *_bwd_context->bwd_desc, eng, fwd_pd));
      _bwd_context->pool_bwd.reset(new pooling_backward(*_bwd_context->bwd_pd));
      // get memory
      _bwd_context->in_diff_mem.reset(
          new memory(_bwd_context->bwd_pd.get()->diff_dst_desc(), eng));
      _bwd_context->ws_mem.reset(new memory(tensors[ws_name]->desc(), eng));
      _bwd_context->out_diff_mem.reset(
          new memory(_bwd_context->bwd_pd.get()->diff_src_desc(), eng));
      tensors[out_diff_name]->init(_bwd_context->bwd_pd.get()->diff_src_desc(),
                                   dev);
      timings[time_name]["create"] = get_elapsed_ms(time_create);
      dev->memory_used += _bwd_context->get_memory_used();
    }
    // reorders
    auto s = dev->get_stream(0);
    timings[time_name][in_diff_name] =
        maybe_do_reorder(tensors[in_diff_name]->get_memory(),
                         *_bwd_context->in_diff_mem, s, measure_time);
    timings[time_name][ws_name] = maybe_do_reorder(
        tensors[ws_name]->get_memory(), *_bwd_context->ws_mem, s, measure_time);
    // execute
    auto args = unordered_map<int, memory>(
        {{DNNL_ARG_DIFF_DST, *_bwd_context->in_diff_mem},
         {DNNL_ARG_DIFF_SRC, *_bwd_context->out_diff_mem},
         {DNNL_ARG_WORKSPACE, *_bwd_context->ws_mem}});
    auto time_exe = get_time();
    _bwd_context->pool_bwd->execute(s, args);
    s.wait();
    if (measure_time) {
      timings[time_name]["exe"] = get_elapsed_ms(time_exe);
      timings[time_name]["total"] = get_elapsed_ms(begin);
    }
  }
  memory::dims stride;
  memory::dims kernel;
  memory::dims padding_l, padding_r;
  algorithm algo;
  shared_ptr<PoolFwdContext> _fwd_context;
  shared_ptr<PoolBwdContext> _bwd_context;
};
#endif
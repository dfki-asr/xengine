#ifndef XENGINE_OP_GEMM_HPP
#define XENGINE_OP_GEMM_HPP

#include "../operator.hpp"

struct GemmFwdContext {
  shared_ptr<memory> src_mem;
  shared_ptr<memory> weights_mem;
  shared_ptr<memory> bias_mem;
  shared_ptr<memory> dst_mem;
  shared_ptr<inner_product_forward::desc> fwd_desc;
  shared_ptr<inner_product_forward::primitive_desc> fwd_pd;
  shared_ptr<inner_product_forward> gemm_fwd;

  GemmFwdContext()
      : src_mem(nullptr), weights_mem(nullptr), bias_mem(nullptr),
        dst_mem(nullptr), fwd_desc(nullptr), fwd_pd(nullptr),
        gemm_fwd(nullptr) {}

  ~GemmFwdContext() {
    src_mem.reset();
    weights_mem.reset();
    bias_mem.reset();
    dst_mem.reset();
    fwd_desc.reset();
    fwd_pd.reset();
    gemm_fwd.reset();
  }

  size_t get_memory_used() {
    size_t memory_used_bytes = 0;
    if (src_mem != nullptr)
      memory_used_bytes += src_mem->get_desc().get_size();
    if (weights_mem != nullptr)
      memory_used_bytes += weights_mem->get_desc().get_size();
    if (bias_mem != nullptr)
      memory_used_bytes += bias_mem->get_desc().get_size();
    if (dst_mem != nullptr)
      memory_used_bytes += dst_mem->get_desc().get_size();
    return memory_used_bytes;
  }
};

struct GemmBwdContext {
  shared_ptr<memory> in_diff_mem;
  shared_ptr<memory> src_mem;
  shared_ptr<memory> weights_mem;
  shared_ptr<memory> bias_mem;
  shared_ptr<memory> in_diff_d_mem;
  shared_ptr<memory> out_diff_mem;
  shared_ptr<memory> w_diff_mem;
  shared_ptr<memory> b_diff_mem;
  shared_ptr<inner_product_backward_weights::desc> bwd_w_desc;
  shared_ptr<inner_product_backward_data::desc> bwd_d_desc;
  shared_ptr<inner_product_backward_weights::primitive_desc> bwd_w_pd;
  shared_ptr<inner_product_backward_data::primitive_desc> bwd_d_pd;
  shared_ptr<inner_product_backward_weights> gemm_bwd_weights;
  shared_ptr<inner_product_backward_data> gemm_bwd_data;

  GemmBwdContext()
      : in_diff_mem(nullptr), src_mem(nullptr), weights_mem(nullptr),
        bias_mem(nullptr), in_diff_d_mem(nullptr), out_diff_mem(nullptr),
        w_diff_mem(nullptr), b_diff_mem(nullptr), bwd_w_desc(nullptr),
        bwd_d_desc(nullptr), bwd_w_pd(nullptr), bwd_d_pd(nullptr),
        gemm_bwd_weights(nullptr), gemm_bwd_data(nullptr) {}

  ~GemmBwdContext() {
    in_diff_mem.reset();
    src_mem.reset();
    weights_mem.reset();
    bias_mem.reset();
    in_diff_d_mem.reset();
    out_diff_mem.reset();
    w_diff_mem.reset();
    b_diff_mem.reset();
    bwd_w_desc.reset();
    bwd_d_desc.reset();
    bwd_w_pd.reset();
    bwd_d_pd.reset();
    gemm_bwd_weights.reset();
    gemm_bwd_data.reset();
  }

  size_t get_memory_used() {
    size_t memory_used_bytes = 0;
    if (in_diff_mem != nullptr)
      memory_used_bytes += in_diff_mem->get_desc().get_size();
    if (src_mem != nullptr)
      memory_used_bytes += src_mem->get_desc().get_size();
    if (weights_mem != nullptr)
      memory_used_bytes += weights_mem->get_desc().get_size();
    if (bias_mem != nullptr)
      memory_used_bytes += bias_mem->get_desc().get_size();
    if (in_diff_d_mem != nullptr)
      memory_used_bytes += in_diff_d_mem->get_desc().get_size();
    if (out_diff_mem != nullptr)
      memory_used_bytes += out_diff_mem->get_desc().get_size();
    if (w_diff_mem != nullptr)
      memory_used_bytes += w_diff_mem->get_desc().get_size();
    if (b_diff_mem != nullptr)
      memory_used_bytes += b_diff_mem->get_desc().get_size();
    return memory_used_bytes;
  }
};

class Gemm : public Operator_With_Weights {
public:
  Gemm(string n, vector<string> i, vector<string> o,
       unordered_map<string, shared_ptr<Tensor>> &tensors, int training)
      : Operator_With_Weights(n, "Gemm", i, o, tensors, training) {
    _fwd_context = nullptr;
    _bwd_context = nullptr;
    init(tensors);
  }
  ~Gemm() {
    reset_fwd_primitives();
    reset_bwd_primitives();
  }
  void reset_fwd_primitives() {
    if (_f_device != nullptr && _fwd_context != nullptr &&
        _track_only_tensor_memory == 0) {
      _f_device->memory_used -= _fwd_context->get_memory_used();
    }
    _fwd_context.reset();
  }
  void reset_bwd_primitives() {
    if (_b_device != nullptr && _bwd_context != nullptr &&
        _track_only_tensor_memory == 0) {
      _b_device->memory_used -= _bwd_context->get_memory_used();
    }
    _bwd_context.reset();
  }

  void forward(shared_ptr<Device> dev,
               unordered_map<string, shared_ptr<Tensor>> &tensors,
               memory::format_tag outputTag, const int measure_time) {
    _f_device = dev;
    auto begin = get_time();
    auto eng = dev->get_engine();
    auto src_name = _f_op.input.at(0);
    auto w_name = _f_op.input.at(1);
    auto out_name = _f_op.output.at(0);
    auto src_dims = tensors[_f_op.input.at(0)]->dims();
    auto w_dims = tensors[w_name]->dims();
    auto b_md = has_bias() ? getDesc({w_dims.at(0)}, memory::format_tag::x)
                           : memory::desc();
    auto dst_dims = memory::dims({src_dims.at(0), w_dims.at(0)});
    if (src_dims.size() > w_dims.size()) {
      src_dims = {src_dims.begin(), src_dims.begin() + w_dims.size()};
    }
    auto time_name = getForwardTimeName(eng);
    auto s = dev->get_stream(0);
    if (_fwd_context == nullptr) {
      auto time_create = get_time();
      _fwd_context.reset(new GemmFwdContext());
      _fwd_context->fwd_desc.reset(new inner_product_forward::desc(
          {getMode(training), getDesc(src_dims), getDesc(w_dims), b_md,
           getDesc(dst_dims, outputTag)}));
      _fwd_context->fwd_pd.reset(new inner_product_forward::primitive_desc(
          *_fwd_context->fwd_desc, eng));
      _fwd_context->gemm_fwd.reset(
          new inner_product_forward(*_fwd_context->fwd_pd));
      // get memory
      _fwd_context->src_mem.reset(
          new memory(_fwd_context->fwd_pd.get()->src_desc(), eng));
      _fwd_context->weights_mem.reset(
          new memory(_fwd_context->fwd_pd.get()->weights_desc(), eng));
      _fwd_context->dst_mem.reset(
          new memory(_fwd_context->fwd_pd.get()->dst_desc(), eng));
      tensors[out_name]->init(_fwd_context->fwd_pd.get()->dst_desc(), dev);
      timings[time_name][w_name] =
          maybe_do_reorder(tensors[w_name]->get_memory(),
                           *_fwd_context->weights_mem, s, measure_time);
      if (has_bias()) {
        _fwd_context->bias_mem.reset(new memory(b_md, eng));
        auto b_name = _f_op.input.at(2);
        timings[time_name][b_name] =
            maybe_do_reorder(tensors[b_name]->get_memory(),
                             *_fwd_context->bias_mem, s, measure_time);
      }
      timings[time_name]["create"] = get_elapsed_ms(time_create);
      if (_track_only_tensor_memory == 0) {
        dev->memory_used += _fwd_context->get_memory_used();
      }
    }
    // reorders
    timings[time_name][src_name] =
        maybe_do_reorder(tensors[src_name]->get_memory(),
                         *_fwd_context->src_mem, s, measure_time);
    // execute
    auto args = unordered_map<int, memory>(
        {{DNNL_ARG_SRC, *_fwd_context->src_mem},
         {DNNL_ARG_WEIGHTS, *_fwd_context->weights_mem},
         {DNNL_ARG_DST, *_fwd_context->dst_mem}});
    if (has_bias()) {
      args.insert({DNNL_ARG_BIAS, *_fwd_context->bias_mem});
    }
    auto time_exe = get_time();
    _fwd_context->gemm_fwd->execute(s, args);
    s.wait();
    if (measure_time) {
      timings[time_name]["exe"] = get_elapsed_ms(time_exe);
      timings[time_name]["total"] = get_elapsed_ms(begin);
    }
  }

  void backward(shared_ptr<Device> dev,
                unordered_map<string, shared_ptr<Tensor>> &tensors,
                memory::format_tag outputTag, const int measure_time) {
    _b_device = dev;
    auto begin = get_time();
    auto eng = dev->get_engine();
    auto src_name = _b_op.input.at(1);
    auto w_name = _b_op.input.at(2);
    auto in_diff_name = _b_op.input.at(0);
    auto w_diff_name = _b_op.output.at(1);
    auto out_diff_name = _b_op.output.at(0);
    auto src_dims = tensors[src_name]->dims();
    auto src_md = getDesc(src_dims);
    auto w_dims = tensors[w_name]->dims();
    auto w_md = getDesc(w_dims);
    auto b_md = has_bias() ? getDesc({w_dims.at(0)}, memory::format_tag::x)
                           : memory::desc();
    auto dst_dims = memory::dims({src_dims.at(0), w_dims.at(0)});
    auto dst_md = getDesc(dst_dims, outputTag);
    auto time_name = getBackwardTimeName(eng);
    auto s = dev->get_stream(0);
    if (_bwd_context == nullptr) {
      auto time_create = get_time();
      _bwd_context.reset(new GemmBwdContext());
      _bwd_context->bwd_w_desc.reset(new inner_product_backward_weights::desc(
          {src_md, w_md, b_md, dst_md}));
      int reuse_fwd_pd = 0;
      if (_fwd_context != nullptr) {
        reuse_fwd_pd = _fwd_context->fwd_pd->get_engine() == eng ? 1 : 0;
      }
      auto fwd_pd =
          reuse_fwd_pd
              ? *_fwd_context->fwd_pd
              : inner_product_forward::primitive_desc(
                    {prop_kind::forward_training, src_md, w_md, b_md, dst_md},
                    eng);
      _bwd_context->bwd_w_pd.reset(
          new inner_product_backward_weights::primitive_desc(
              *_bwd_context->bwd_w_desc, eng, fwd_pd));
      _bwd_context->bwd_d_desc.reset(
          new inner_product_backward_data::desc({src_md, w_md, dst_md}));
      _bwd_context->bwd_d_pd.reset(
          new inner_product_backward_data::primitive_desc(
              *_bwd_context->bwd_d_desc, eng, fwd_pd));
      _bwd_context->gemm_bwd_weights.reset(
          new inner_product_backward_weights(*_bwd_context->bwd_w_pd));
      _bwd_context->gemm_bwd_data.reset(
          new inner_product_backward_data(*_bwd_context->bwd_d_pd));
      // get memory
      _bwd_context->in_diff_mem.reset(
          new memory(_bwd_context->bwd_w_pd.get()->diff_dst_desc(), eng));
      _bwd_context->src_mem.reset(
          new memory(_bwd_context->bwd_w_pd.get()->src_desc(), eng));
      _bwd_context->w_diff_mem.reset(
          new memory(_bwd_context->bwd_w_pd.get()->diff_weights_desc(), eng));
      tensors[w_diff_name]->init(
          _bwd_context->bwd_w_pd.get()->diff_weights_desc(), dev);
      _bwd_context->in_diff_d_mem.reset(
          new memory(_bwd_context->bwd_d_pd.get()->diff_dst_desc(), eng));
      _bwd_context->weights_mem.reset(
          new memory(_bwd_context->bwd_d_pd.get()->weights_desc(), eng));
      _bwd_context->out_diff_mem.reset(
          new memory(_bwd_context->bwd_d_pd.get()->diff_src_desc(), eng));
      tensors[out_diff_name]->init(
          _bwd_context->bwd_d_pd.get()->diff_src_desc(), dev);
      timings[time_name][w_name] =
          maybe_do_reorder(tensors[w_name]->get_memory(),
                           *_bwd_context->weights_mem, s, measure_time);
      if (has_bias()) {
        auto b_diff_name = _b_op.output.at(2);
        auto w_dims = tensors[w_diff_name]->dims();
        auto b_md = getDesc({w_dims.at(0)}, memory::format_tag::x);
        _bwd_context->b_diff_mem.reset(new memory(b_md, eng));
        tensors[b_diff_name]->init(b_md, dev);
      }
      timings[time_name]["create"] = get_elapsed_ms(time_create);
      if (_track_only_tensor_memory == 0) {
        dev->memory_used += _bwd_context->get_memory_used();
      }
    }
    /*********************** Backward Weights *****************************/
    // reorders
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
         {DNNL_ARG_DIFF_WEIGHTS, *_bwd_context->w_diff_mem}});
    if (has_bias()) {
      args.insert({DNNL_ARG_DIFF_BIAS, *_bwd_context->b_diff_mem});
    }
    auto time_exe = get_time();
    _bwd_context->gemm_bwd_weights->execute(s, args);
    s.wait();
    if (measure_time) {
      timings[time_name]["exe"] = get_elapsed_ms(time_exe);
    }
    /*********************** Backward Data *****************************/
    // reorders
    timings[time_name][in_diff_name] +=
        maybe_do_reorder(*_bwd_context->in_diff_mem,
                         *_bwd_context->in_diff_d_mem, s, measure_time);
    // execute
    args = unordered_map<int, memory>(
        {{DNNL_ARG_DIFF_DST, *_bwd_context->in_diff_d_mem},
         {DNNL_ARG_DIFF_SRC, *_bwd_context->out_diff_mem},
         {DNNL_ARG_WEIGHTS, *_bwd_context->weights_mem}});
    time_exe = get_time();
    _bwd_context->gemm_bwd_data->execute(s, args);
    s.wait();
    if (measure_time) {
      timings[time_name]["exe"] += get_elapsed_ms(time_exe);
      timings[time_name]["total"] = get_elapsed_ms(begin);
    }
  }
  shared_ptr<GemmFwdContext> _fwd_context;
  shared_ptr<GemmBwdContext> _bwd_context;
};
#endif
#ifndef XENGINE_OP_CONVTRANSPOSE_HPP
#define XENGINE_OP_CONVTRANSPOSE_HPP

#include "../operator.hpp"

struct ConvTFwdContext {
  shared_ptr<memory> src_mem;
  shared_ptr<memory> in_diff_mem;
  shared_ptr<memory> weights_mem;
  shared_ptr<memory> bias_mem;
  shared_ptr<memory> dst_mem;
  shared_ptr<convolution_forward::desc> fwd_f_desc;
  shared_ptr<convolution_forward::primitive_desc> fwd_f_pd;
  shared_ptr<convolution_backward_data::desc> fwd_b_desc;
  shared_ptr<convolution_backward_data::primitive_desc> fwd_b_pd;
  shared_ptr<convolution_backward_data> convT_fwd;

  ConvTFwdContext()
      : src_mem(nullptr), in_diff_mem(nullptr), weights_mem(nullptr),
        bias_mem(nullptr), dst_mem(nullptr), fwd_f_desc(nullptr),
        fwd_f_pd(nullptr), fwd_b_desc(nullptr), fwd_b_pd(nullptr),
        convT_fwd(nullptr) {}

  ~ConvTFwdContext() {
    src_mem.reset();
    in_diff_mem.reset();
    weights_mem.reset();
    bias_mem.reset();
    dst_mem.reset();
    fwd_f_desc.reset();
    fwd_f_pd.reset();
    fwd_b_desc.reset();
    fwd_b_pd.reset();
    convT_fwd.reset();
  }

  size_t get_memory_used() {
    size_t memory_used_bytes = 0;
    if (src_mem != nullptr)
      memory_used_bytes += src_mem->get_desc().get_size();
    if (in_diff_mem != nullptr)
      memory_used_bytes += in_diff_mem->get_desc().get_size();
    if (weights_mem != nullptr)
      memory_used_bytes += weights_mem->get_desc().get_size();
    if (bias_mem != nullptr)
      memory_used_bytes += bias_mem->get_desc().get_size();
    if (dst_mem != nullptr)
      memory_used_bytes += dst_mem->get_desc().get_size();
    return memory_used_bytes;
  }
};

struct ConvTBwdContext {
  shared_ptr<memory> in_diff_mem;
  shared_ptr<memory> src_mem;
  shared_ptr<memory> weights_mem;
  shared_ptr<memory> bias_mem;
  shared_ptr<memory> in_diff_d_mem;
  shared_ptr<memory> out_diff_mem;
  shared_ptr<memory> w_diff_mem;
  shared_ptr<memory> b_diff_mem;
  shared_ptr<convolution_backward_weights::desc> bwd_w_desc;
  shared_ptr<convolution_forward::desc> bwd_d_desc;
  shared_ptr<convolution_backward_weights::primitive_desc> bwd_w_pd;
  shared_ptr<convolution_forward::primitive_desc> bwd_d_pd;
  shared_ptr<convolution_backward_weights> convT_bwd_weights;
  shared_ptr<convolution_forward> convT_bwd_data;

  ConvTBwdContext()
      : in_diff_mem(nullptr), src_mem(nullptr), weights_mem(nullptr),
        bias_mem(nullptr), in_diff_d_mem(nullptr), out_diff_mem(nullptr),
        w_diff_mem(nullptr), b_diff_mem(nullptr), bwd_w_desc(nullptr),
        bwd_d_desc(nullptr), bwd_w_pd(nullptr), bwd_d_pd(nullptr),
        convT_bwd_weights(nullptr), convT_bwd_data(nullptr) {}

  ~ConvTBwdContext() {
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
    convT_bwd_weights.reset();
    convT_bwd_data.reset();
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

class ConvTranspose : public Operator_With_Weights {
public:
  ConvTranspose(string n, vector<string> i, vector<string> o, memory::dims s,
                memory::dims k, memory::dims p,
                unordered_map<string, shared_ptr<Tensor>> &tensors,
                int training)
      : Operator_With_Weights(n, "ConvTranspose", i, o, tensors, training) {
    stride = s;
    kernel = k;
    padding_l = {p.begin(), p.begin() + k.size()};
    padding_r = {p.begin() + k.size(), p.end()};
    algo = algorithm::convolution_auto;
    _fwd_context = nullptr;
    init(tensors);
  }
  ~ConvTranspose() { reset_fwd_primitives(); }
  void reset_fwd_primitives() {
    if (_f_device != nullptr && _fwd_context != nullptr &&
        _track_only_tensor_memory == 0) {
      _f_device->memory_used -= _fwd_context->get_memory_used();
    }
    _fwd_context.reset();
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
    auto src_md = getDesc(tensors[src_name]->dims());
    auto dst_md = getDesc(tensors[out_name]->dims());
    auto w_md = getDesc(tensors[w_name]->dims());
    auto time_name = getForwardTimeName(dev->name);
    auto s = dev->get_stream(0);
    if (_fwd_context == nullptr) {
      auto time_create = get_time();
      _fwd_context.reset(new ConvTFwdContext());
      _fwd_context->fwd_f_desc.reset(new convolution_forward::desc(
          {prop_kind::forward_inference, algo, dst_md, w_md, src_md, stride,
           padding_l, padding_r}));
      _fwd_context->fwd_f_pd.reset(new convolution_forward::primitive_desc(
          *_fwd_context->fwd_f_desc, eng));
      _fwd_context->fwd_b_desc.reset(new convolution_backward_data::desc(
          {algo, dst_md, w_md, src_md, stride, padding_l, padding_r}));
      _fwd_context->fwd_b_pd.reset(
          new convolution_backward_data::primitive_desc(
              *_fwd_context->fwd_b_desc, eng, *_fwd_context->fwd_f_pd));
      _fwd_context->convT_fwd.reset(
          new convolution_backward_data(*_fwd_context->fwd_b_pd));
      // get memory
      _fwd_context->in_diff_mem.reset(
          new memory(_fwd_context->fwd_b_pd.get()->diff_dst_desc(), eng));
      _fwd_context->src_mem.reset(
          new memory(_fwd_context->fwd_f_pd.get()->dst_desc(), eng));
      _fwd_context->weights_mem.reset(
          new memory(_fwd_context->fwd_b_pd.get()->weights_desc(), eng));
      _fwd_context->dst_mem.reset(
          new memory(_fwd_context->fwd_f_pd.get()->src_desc(), eng));
      tensors[out_name]->init(_fwd_context->fwd_f_pd.get()->src_desc(), dev);
      timings[time_name][w_name] =
          maybe_do_reorder(tensors[w_name]->get_memory(),
                           *_fwd_context->weights_mem, s, measure_time);
      timings[time_name]["create"] = get_elapsed_ms(time_create);
      if (_track_only_tensor_memory == 0) {
        dev->memory_used += _fwd_context->get_memory_used();
      }
    }
    // reorders
    timings[time_name][src_name] =
        maybe_do_reorder(tensors[src_name]->get_memory(),
                         *_fwd_context->in_diff_mem, s, measure_time);
    if (has_bias()) {
      throw runtime_error("ConvTranspose with bias not yet implemented!");
    }
    // execute
    auto time_exe = get_time();
    auto args = unordered_map<int, memory>(
        {{DNNL_ARG_DIFF_DST, *_fwd_context->in_diff_mem},
         {DNNL_ARG_DIFF_SRC, *_fwd_context->dst_mem},
         {DNNL_ARG_WEIGHTS, *_fwd_context->weights_mem}});
    _fwd_context->convT_fwd->execute(s, args);
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
    auto src_md = getDesc(tensors[src_name]->dims());
    auto w_dims = tensors[w_name]->dims();
    auto w_md = getDesc(w_dims);
    if (has_bias()) {
      throw runtime_error("ConvTranspose with bias not yet implemented!");
    }
    auto dst_md = getDesc(tensors[in_diff_name]->dims());
    auto out_diff_md = getDesc(tensors[out_diff_name]->dims());
    auto time_name = getBackwardTimeName(dev->name);
    auto s = dev->get_stream(0);
    if (_bwd_context == nullptr) {
      auto time_create = get_time();
      _bwd_context.reset(new ConvTBwdContext());
      // backward data
      _bwd_context->bwd_d_desc.reset(new convolution_forward::desc(
          {getMode(training), algo, dst_md, w_md, out_diff_md, stride,
           padding_l, padding_r}));
      _bwd_context->bwd_d_pd.reset(new convolution_forward::primitive_desc(
          *_bwd_context->bwd_d_desc, eng));
      _bwd_context->convT_bwd_data.reset(
          new convolution_forward(*_bwd_context->bwd_d_pd));
      // backward W
      _bwd_context->bwd_w_desc.reset(new convolution_backward_weights::desc(
          {algo, dst_md, w_md, src_md, stride, padding_l, padding_r}));
      int reuse_fwd_pd = 0;
      if (_fwd_context != nullptr) {
        reuse_fwd_pd = _fwd_context->fwd_f_pd->get_engine() == eng ? 1 : 0;
      }
      auto fwd_pd = reuse_fwd_pd
                        ? *_fwd_context->fwd_f_pd
                        : convolution_forward::primitive_desc(
                              {prop_kind::forward_training, algo, dst_md, w_md,
                               src_md, stride, padding_l, padding_r},
                              eng);
      _bwd_context->bwd_w_pd.reset(
          new convolution_backward_weights::primitive_desc(
              *_bwd_context->bwd_w_desc, eng, fwd_pd));
      _bwd_context->convT_bwd_weights.reset(
          new convolution_backward_weights(*_bwd_context->bwd_w_pd));
      // get memory
      _bwd_context->in_diff_mem.reset(
          new memory(_bwd_context->bwd_w_pd.get()->src_desc(), eng));
      _bwd_context->src_mem.reset(
          new memory(_bwd_context->bwd_w_pd.get()->dst_desc(), eng));
      _bwd_context->w_diff_mem.reset(
          new memory(_bwd_context->bwd_w_pd.get()->diff_weights_desc(), eng));
      tensors[w_diff_name]->init(
          _bwd_context->bwd_w_pd.get()->diff_weights_desc(), dev);
      _bwd_context->weights_mem.reset(
          new memory(_bwd_context->bwd_d_pd.get()->weights_desc(), eng));
      _bwd_context->out_diff_mem.reset(
          new memory(_bwd_context->bwd_d_pd.get()->dst_desc(), eng));
      tensors[out_diff_name]->init(_bwd_context->bwd_d_pd.get()->dst_desc(),
                                   dev);
      timings[time_name][w_name] =
          maybe_do_reorder(tensors[w_name]->get_memory(),
                           *_bwd_context->weights_mem, s, measure_time);
      timings[time_name]["create"] = get_elapsed_ms(time_create);
      if (_track_only_tensor_memory == 0) {
        dev->memory_used += _bwd_context->get_memory_used();
      }
    }
    /*********************** Backward Data *****************************/
    // reorders
    timings[time_name][in_diff_name] =
        maybe_do_reorder(tensors[in_diff_name]->get_memory(),
                         *_bwd_context->in_diff_mem, s, measure_time);
    // execute
    auto args = unordered_map<int, memory>(
        {{DNNL_ARG_SRC, *_bwd_context->in_diff_mem},
         {DNNL_ARG_WEIGHTS, *_bwd_context->weights_mem},
         {DNNL_ARG_DST, *_bwd_context->out_diff_mem}});
    auto time_exe = get_time();
    _bwd_context->convT_bwd_data->execute(s, args);
    s.wait();
    if (measure_time) {
      timings[time_name]["exe"] = get_elapsed_ms(time_exe);
    }
    /*********************** Backward Weights *****************************/
    // reorders
    timings[time_name][in_diff_name] =
        maybe_do_reorder(tensors[in_diff_name]->get_memory(),
                         *_bwd_context->in_diff_mem, s, measure_time);

    auto tmp_mem = memory(_bwd_context->bwd_w_pd.get()->diff_dst_desc(), eng);
    // execute
    args = unordered_map<int, memory>(
        {{DNNL_ARG_SRC, *_bwd_context->in_diff_mem},
         {DNNL_ARG_DIFF_DST, tmp_mem},
         {DNNL_ARG_DIFF_WEIGHTS, *_bwd_context->w_diff_mem}});
    time_exe = get_time();
    _bwd_context->convT_bwd_weights->execute(s, args);
    s.wait();
    if (measure_time) {
      timings[time_name]["exe"] += get_elapsed_ms(time_exe);
      timings[time_name]["total"] = get_elapsed_ms(begin);
    }
  }
  memory::dims stride;
  memory::dims kernel;
  memory::dims padding_l, padding_r;
  algorithm algo;
  shared_ptr<ConvTFwdContext> _fwd_context;
  shared_ptr<ConvTBwdContext> _bwd_context;
};
#endif
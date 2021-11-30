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
};
class ConvTranspose : public Operator_With_Weights {
public:
  ConvTranspose(string n, vector<string> i, vector<string> o, memory::dims s,
                memory::dims k, memory::dims p,
                unordered_map<string, unique_ptr<Tensor>> &tensors,
                int training)
      : Operator_With_Weights(n, "ConvTranspose", i, o, tensors, training) {
    stride = s;
    kernel = k;
    padding = p;
    algo = algorithm::convolution_auto;
    _fwd_context = nullptr;
    init(tensors);
  }
  ~ConvTranspose() { reset_fwd_primitives(); }
  void reset_fwd_primitives() { _fwd_context.reset(); }

  void forward(Device &dev, unordered_map<string, unique_ptr<Tensor>> &tensors,
               memory::format_tag outputTag, const int measure_time) {
    auto begin = get_time();
    auto eng = dev.get_engine();
    auto src_name = _f_op.input.at(0);
    auto w_name = _f_op.input.at(1);
    auto out_name = _f_op.output.at(0);
    auto src_md = getDesc(tensors[src_name]->dims());
    auto dst_md = getDesc(tensors[out_name]->dims());
    auto w_md = getDesc(tensors[w_name]->dims());
    auto time_name = getForwardTimeName(eng);
    auto s = stream(eng);
    if (_fwd_context == nullptr) {
      auto time_create = get_time();
      _fwd_context.reset(new ConvTFwdContext());
      _fwd_context->fwd_f_desc.reset(new convolution_forward::desc(
          {prop_kind::forward_inference, algo, dst_md, w_md, src_md, stride,
           padding, padding}));
      _fwd_context->fwd_f_pd.reset(new convolution_forward::primitive_desc(
          *_fwd_context->fwd_f_desc, eng));
      _fwd_context->fwd_b_desc.reset(new convolution_backward_data::desc(
          {algo, dst_md, w_md, src_md, stride, padding, padding}));
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
      tensors[out_name]->init(_fwd_context->fwd_f_pd.get()->src_desc(), eng);
      timings[time_name][w_name] =
          maybe_do_reorder(tensors[w_name]->get_memory(),
                           *_fwd_context->weights_mem, s, measure_time);
      timings[time_name]["create"] = get_elapsed_ms(time_create);
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

  void backward(Device &dev, unordered_map<string, unique_ptr<Tensor>> &tensors,
                memory::format_tag outputTag, const int measure_time) {
    throw runtime_error("convTranspose backward not yet implemented!");
  }

  memory::dims stride;
  memory::dims kernel;
  memory::dims padding;
  algorithm algo;
  shared_ptr<ConvTFwdContext> _fwd_context;
};
#endif
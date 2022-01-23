#ifndef XENGINE_OP_BATCH_NORMALIZATION_HPP
#define XENGINE_OP_BATCH_NORMALIZATION_HPP

#include "../operator.hpp"

using namespace dnnl;

struct BNFwdContext {
  shared_ptr<memory> src_mem;
  shared_ptr<memory> gamma_beta_mem;
  shared_ptr<memory> mean_mem;
  shared_ptr<memory> var_mem;
  shared_ptr<memory> dst_mem;
  shared_ptr<batch_normalization_forward::desc> fwd_desc;
  shared_ptr<batch_normalization_forward::primitive_desc> fwd_pd;
  shared_ptr<batch_normalization_forward> batchnorm_fwd;

  BNFwdContext()
      : src_mem(nullptr), gamma_beta_mem(nullptr), mean_mem(nullptr),
        var_mem(nullptr), dst_mem(nullptr), fwd_desc(nullptr), fwd_pd(nullptr),
        batchnorm_fwd(nullptr) {}

  ~BNFwdContext() {
    src_mem.reset();
    gamma_beta_mem.reset();
    mean_mem.reset();
    var_mem.reset();
    dst_mem.reset();
    fwd_desc.reset();
    fwd_pd.reset();
    batchnorm_fwd.reset();
  }

  size_t get_memory_used() {
    size_t memory_used_bytes = 0;
    if (src_mem != nullptr)
      memory_used_bytes += src_mem->get_desc().get_size();
    if (gamma_beta_mem != nullptr)
      memory_used_bytes += gamma_beta_mem->get_desc().get_size();
    if (mean_mem != nullptr)
      memory_used_bytes += mean_mem->get_desc().get_size();
    if (var_mem != nullptr)
      memory_used_bytes += var_mem->get_desc().get_size();
    if (dst_mem != nullptr)
      memory_used_bytes += dst_mem->get_desc().get_size();
    return memory_used_bytes;
  }
};

struct BNBwdContext {
  shared_ptr<memory> in_diff_mem;
  shared_ptr<memory> src_mem;
  shared_ptr<memory> gamma_beta_mem;
  shared_ptr<memory> mean_mem;
  shared_ptr<memory> var_mem;
  shared_ptr<memory> out_diff_mem;
  shared_ptr<memory> gamma_diff_mem;
  shared_ptr<memory> beta_diff_mem;
  shared_ptr<memory> gamma_beta_diff_mem;
  shared_ptr<batch_normalization_backward::desc> bwd_desc;
  shared_ptr<batch_normalization_backward::primitive_desc> bwd_pd;
  shared_ptr<batch_normalization_backward> batchnorm_bwd;

  BNBwdContext()
      : in_diff_mem(nullptr), src_mem(nullptr), gamma_beta_mem(nullptr),
        mean_mem(nullptr), var_mem(nullptr), out_diff_mem(nullptr),
        gamma_diff_mem(nullptr), beta_diff_mem(nullptr),
        gamma_beta_diff_mem(nullptr), bwd_desc(nullptr), bwd_pd(nullptr),
        batchnorm_bwd(nullptr) {}

  ~BNBwdContext() {
    in_diff_mem.reset();
    src_mem.reset();
    gamma_beta_mem.reset();
    mean_mem.reset();
    var_mem.reset();
    out_diff_mem.reset();
    gamma_diff_mem.reset();
    beta_diff_mem.reset();
    gamma_beta_diff_mem.reset();
    bwd_desc.reset();
    bwd_pd.reset();
    batchnorm_bwd.reset();
  }

  size_t get_memory_used() {
    size_t memory_used_bytes = 0;
    if (in_diff_mem != nullptr)
      memory_used_bytes += in_diff_mem->get_desc().get_size();
    if (src_mem != nullptr)
      memory_used_bytes += src_mem->get_desc().get_size();
    if (gamma_beta_mem != nullptr)
      memory_used_bytes += gamma_beta_mem->get_desc().get_size();
    if (mean_mem != nullptr)
      memory_used_bytes += mean_mem->get_desc().get_size();
    if (var_mem != nullptr)
      memory_used_bytes += var_mem->get_desc().get_size();
    if (out_diff_mem != nullptr)
      memory_used_bytes += out_diff_mem->get_desc().get_size();
    if (gamma_diff_mem != nullptr)
      memory_used_bytes += gamma_diff_mem->get_desc().get_size();
    if (beta_diff_mem != nullptr)
      memory_used_bytes += beta_diff_mem->get_desc().get_size();
    if (gamma_beta_diff_mem != nullptr)
      memory_used_bytes += gamma_beta_diff_mem->get_desc().get_size();
    return memory_used_bytes;
  }
};

class BatchNormalization : public Operator {
public:
  BatchNormalization(string n, vector<string> i, vector<string> o, float e,
                     float m,
                     unordered_map<string, shared_ptr<Tensor>> &tensors,
                     int training)
      : Operator(n, "BatchNormalization", i, o, tensors, training) {
    epsilon = e;
    momentum = m;
    auto b_i = vector<string>{"diff_" + o.at(0)};
    for (auto inp : i) {
      b_i.push_back(inp);
    }
    _f_op = ExecutionOp("fwd_" + n, "fwd", i, o);
    _b_op = ExecutionOp("bwd_" + n, "bwd", b_i,
                        vector<string>{"diff_" + i.at(0), "diff_" + i.at(1),
                                       "diff_" + i.at(2)});
    _fwd_context = nullptr;
    _bwd_context = nullptr;
    init(tensors);
  }
  ~BatchNormalization() {
    reset_fwd_primitives();
    reset_bwd_primitives();
  }
  void reset_fwd_primitives() { _fwd_context.reset(); }
  void reset_bwd_primitives() { _bwd_context.reset(); }

  memory
  prepare_gamma_beta(memory::desc &gamma_beta_md,
                     unordered_map<string, shared_ptr<Tensor>> &tensors) {
    auto gamma_name = _f_op.input.at(1);
    auto beta_name = _f_op.input.at(2);
    auto gamma_beta_dims = gamma_beta_md.dims();
    auto channels = gamma_beta_dims.at(1);
    auto cpu_eng = tensors[gamma_name]->get_memory().get_engine();
    if (cpu_eng.get_kind() != engine::kind::cpu ||
        tensors[beta_name]->get_memory().get_engine().get_kind() !=
            engine::kind::cpu) {
      throw runtime_error("BatchNormalization: gamma and beta must be on CPU!");
    }
    auto s_ = stream(cpu_eng);
    auto cpu_gamma_mem = tensors[gamma_name]->get_memory();
    auto cpu_beta_mem = tensors[beta_name]->get_memory();
    auto cpu_gamma = static_cast<float *>(cpu_gamma_mem.get_data_handle());
    auto cpu_beta = static_cast<float *>(cpu_beta_mem.get_data_handle());
    auto cpu_gamma_beta_data = vector<float>(2 * channels);
    copy_n(cpu_gamma, channels, cpu_gamma_beta_data.data());
    copy_n(cpu_beta, channels, cpu_gamma_beta_data.data() + channels);
    return make_memory(gamma_beta_md, cpu_eng, cpu_gamma_beta_data.data());
  }

  void forward(shared_ptr<Device> dev,
               unordered_map<string, shared_ptr<Tensor>> &tensors,
               memory::format_tag outputTag, const int measure_time) {
    auto begin = get_time();
    auto eng = dev->get_engine();
    if (input.size() < 5) {
      throw runtime_error("BatchNormalization: too less inputs!");
    }
    auto src_name = _f_op.input.at(0);
    auto gamma_name = _f_op.input.at(1);
    auto beta_name = _f_op.input.at(2);
    auto mean_name = _f_op.input.at(3);
    auto var_name = _f_op.input.at(4);
    auto out_name = _f_op.output.at(0);
    auto src_md = tensors[src_name]->desc();
    auto channels = src_md.dims().at(1);
    auto gamma_beta_dims = memory::dims(2, channels);
    auto gamma_beta_md = getDesc(gamma_beta_dims, memory::format_tag::nc);
    auto time_name = getForwardTimeName(eng);
    auto s = dev->get_stream(0);
    if (_fwd_context == nullptr) {
      auto time_create = get_time();
      _fwd_context.reset(new BNFwdContext());
      _fwd_context->fwd_desc.reset(new batch_normalization_forward::desc(
          {getMode(training), src_md, epsilon,
           normalization_flags::use_scale_shift}));
      _fwd_context->fwd_pd.reset(
          new batch_normalization_forward::primitive_desc(
              *_fwd_context->fwd_desc, eng));
      _fwd_context->batchnorm_fwd.reset(
          new batch_normalization_forward(*_fwd_context->fwd_pd));
      // get memory
      _fwd_context->src_mem.reset(
          new memory(_fwd_context->fwd_pd.get()->src_desc(), eng));
      _fwd_context->gamma_beta_mem.reset(new memory(gamma_beta_md, eng));
      _fwd_context->mean_mem.reset(new memory(tensors[mean_name]->desc(), eng));
      _fwd_context->var_mem.reset(new memory(tensors[var_name]->desc(), eng));
      _fwd_context->dst_mem.reset(
          new memory(_fwd_context->fwd_pd.get()->dst_desc(), eng));
      tensors[out_name]->init(_fwd_context->fwd_pd.get()->dst_desc(), dev);
      auto gamma_beta_mem = prepare_gamma_beta(gamma_beta_md, tensors);
      maybe_do_reorder(gamma_beta_mem, *_fwd_context->gamma_beta_mem, s, 0);
      timings[time_name][mean_name] =
          maybe_do_reorder(tensors[mean_name]->get_memory(),
                           *_fwd_context->mean_mem, s, measure_time);
      timings[time_name][var_name] =
          maybe_do_reorder(tensors[var_name]->get_memory(),
                           *_fwd_context->var_mem, s, measure_time);
      timings[time_name]["create"] = get_elapsed_ms(time_create);
      dev->memory_used += _fwd_context->get_memory_used();
    }
    // reorders
    timings[time_name][src_name] =
        maybe_do_reorder(tensors[src_name]->get_memory(),
                         *_fwd_context->src_mem, s, measure_time);
    auto args = unordered_map<int, memory>(
        {{DNNL_ARG_SRC, *_fwd_context->src_mem},
         {DNNL_ARG_SCALE_SHIFT, *_fwd_context->gamma_beta_mem},
         {DNNL_ARG_MEAN, *_fwd_context->mean_mem},
         {DNNL_ARG_VARIANCE, *_fwd_context->var_mem},
         {DNNL_ARG_DST, *_fwd_context->dst_mem}});
    auto time_exe = get_time();
    _fwd_context->batchnorm_fwd->execute(s, args);
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
    auto src_name = _b_op.input.at(1);
    auto gamma_name = _b_op.input.at(2);
    auto beta_name = _b_op.input.at(3);
    auto mean_name = _b_op.input.at(4);
    auto var_name = _b_op.input.at(5);
    auto in_diff_name = _b_op.input.at(0);
    auto out_diff_name = _b_op.output.at(0);
    auto gamma_diff_name = _b_op.output.at(1);
    auto beta_diff_name = _b_op.output.at(2);
    auto src_md = tensors[src_name]->desc();
    auto channels = src_md.dims().at(1);
    auto gamma_beta_dims = memory::dims(2, channels);
    auto gamma_beta_md = getDesc(gamma_beta_dims, memory::format_tag::nc);
    auto gamma_beta_diff_md = getDesc(gamma_beta_dims, memory::format_tag::nc);
    auto time_name = getBackwardTimeName(eng);
    auto s = dev->get_stream(0);
    if (_bwd_context == nullptr) {
      auto time_create = get_time();
      _bwd_context.reset(new BNBwdContext());
      _bwd_context->bwd_desc.reset(new batch_normalization_backward::desc(
          {prop_kind::backward, src_md, src_md, epsilon,
           normalization_flags::use_scale_shift}));
      int reuse_fwd_pd = 0;
      if (_fwd_context != nullptr) {
        reuse_fwd_pd = _fwd_context->fwd_pd->get_engine() == eng ? 1 : 0;
      }
      auto fwd_pd = reuse_fwd_pd
                        ? *_fwd_context->fwd_pd
                        : batch_normalization_forward::primitive_desc(
                              {prop_kind::forward_training, src_md, epsilon,
                               normalization_flags::use_scale_shift},
                              eng);
      _bwd_context->bwd_pd.reset(
          new batch_normalization_backward::primitive_desc(
              *_bwd_context->bwd_desc, eng, fwd_pd));
      _bwd_context->batchnorm_bwd.reset(
          new batch_normalization_backward(*_bwd_context->bwd_pd));
      // get memory
      _bwd_context->in_diff_mem.reset(
          new memory(_bwd_context->bwd_pd.get()->diff_dst_desc(), eng));
      _bwd_context->src_mem.reset(
          new memory(_bwd_context->bwd_pd.get()->src_desc(), eng));
      _bwd_context->gamma_beta_mem.reset(new memory(gamma_beta_md, eng));
      _bwd_context->mean_mem.reset(new memory(tensors[mean_name]->desc(), eng));
      _bwd_context->var_mem.reset(new memory(tensors[var_name]->desc(), eng));
      _bwd_context->gamma_diff_mem.reset(
          new memory(tensors[gamma_name]->desc(), eng));
      tensors[gamma_diff_name]->init(tensors[gamma_name]->desc(), dev);
      _bwd_context->beta_diff_mem.reset(
          new memory(tensors[beta_name]->desc(), eng));
      tensors[beta_diff_name]->init(tensors[beta_name]->desc(), dev);
      _bwd_context->gamma_beta_diff_mem.reset(
          new memory(gamma_beta_diff_md, eng));
      _bwd_context->out_diff_mem.reset(new memory(src_md, eng));
      tensors[out_diff_name]->init(src_md, dev);
      auto gamma_beta_mem = prepare_gamma_beta(gamma_beta_md, tensors);
      maybe_do_reorder(gamma_beta_mem, *_bwd_context->gamma_beta_mem, s, 0);
      timings[time_name][mean_name] =
          maybe_do_reorder(tensors[mean_name]->get_memory(),
                           *_bwd_context->mean_mem, s, measure_time);
      timings[time_name][var_name] =
          maybe_do_reorder(tensors[var_name]->get_memory(),
                           *_bwd_context->var_mem, s, measure_time);
      timings[time_name]["create"] = get_elapsed_ms(time_create);
      dev->memory_used += _bwd_context->get_memory_used();
    }
    //  reorders
    timings[time_name][src_name] =
        maybe_do_reorder(tensors[src_name]->get_memory(),
                         *_bwd_context->src_mem, s, measure_time);
    timings[time_name][in_diff_name] =
        maybe_do_reorder(tensors[in_diff_name]->get_memory(),
                         *_bwd_context->in_diff_mem, s, measure_time);
    // execute
    auto args = unordered_map<int, memory>(
        {{DNNL_ARG_SRC, *_bwd_context->src_mem},
         {DNNL_ARG_MEAN, *_bwd_context->mean_mem},
         {DNNL_ARG_VARIANCE, *_bwd_context->var_mem},
         {DNNL_ARG_DIFF_DST, *_bwd_context->in_diff_mem},
         {DNNL_ARG_SCALE_SHIFT, *_bwd_context->gamma_beta_mem},
         {DNNL_ARG_DIFF_SRC, *_bwd_context->out_diff_mem},
         {DNNL_ARG_DIFF_SCALE_SHIFT, *_bwd_context->gamma_beta_diff_mem}});
    auto time_exe = get_time();
    _bwd_context->batchnorm_bwd->execute(s, args);
    s.wait();
    if (measure_time) {
      timings[time_name]["exe"] = get_elapsed_ms(time_exe);
      timings[time_name]["total"] = get_elapsed_ms(begin);
    }
  }
  float epsilon;
  float momentum;
  shared_ptr<BNFwdContext> _fwd_context;
  shared_ptr<BNBwdContext> _bwd_context;
};
#endif
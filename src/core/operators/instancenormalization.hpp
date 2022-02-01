#ifndef XENGINE_OP_INSTANCE_NORMALIZATION_HPP
#define XENGINE_OP_INSTANCE_NORMALIZATION_HPP

#include "batchnormalization.hpp"

using namespace dnnl;

struct INFwdContext {
  shared_ptr<memory> src_mem;
  shared_ptr<memory> gamma_mem;
  shared_ptr<memory> beta_mem;
  shared_ptr<memory> mean_mem;
  shared_ptr<memory> var_mem;
  shared_ptr<memory> weights_mem;
  shared_ptr<memory> dst_mem;
  shared_ptr<batch_normalization_forward::desc> fwd_desc;
  shared_ptr<batch_normalization_forward::primitive_desc> fwd_pd;
  shared_ptr<batch_normalization_forward> instancenorm_fwd;

  INFwdContext()
      : src_mem(nullptr), gamma_mem(nullptr), beta_mem(nullptr),
        mean_mem(nullptr), var_mem(nullptr), weights_mem(nullptr),
        dst_mem(nullptr), fwd_desc(nullptr), fwd_pd(nullptr),
        instancenorm_fwd(nullptr) {}

  ~INFwdContext() {
    src_mem.reset();
    gamma_mem.reset();
    beta_mem.reset();
    mean_mem.reset();
    var_mem.reset();
    weights_mem.reset();
    dst_mem.reset();
    fwd_desc.reset();
    fwd_pd.reset();
    instancenorm_fwd.reset();
  }

  size_t get_memory_used() {
    size_t memory_used_bytes = 0;
    if (src_mem != nullptr)
      memory_used_bytes += src_mem->get_desc().get_size();
    if (gamma_mem != nullptr)
      memory_used_bytes += gamma_mem->get_desc().get_size();
    if (beta_mem != nullptr)
      memory_used_bytes += beta_mem->get_desc().get_size();
    if (mean_mem != nullptr)
      memory_used_bytes += mean_mem->get_desc().get_size();
    if (var_mem != nullptr)
      memory_used_bytes += var_mem->get_desc().get_size();
    if (weights_mem != nullptr)
      memory_used_bytes += weights_mem->get_desc().get_size();
    if (dst_mem != nullptr)
      memory_used_bytes += dst_mem->get_desc().get_size();
    return memory_used_bytes;
  }
};

struct INBwdContext {
  shared_ptr<memory> in_diff_mem;
  shared_ptr<memory> src_mem;
  shared_ptr<memory> gamma_mem;
  shared_ptr<memory> beta_mem;
  shared_ptr<memory> mean_mem;
  shared_ptr<memory> var_mem;
  shared_ptr<memory> weights_mem;
  shared_ptr<memory> gamma_diff_mem;
  shared_ptr<memory> out_diff_mem;
  shared_ptr<batch_normalization_backward::desc> bwd_desc;
  shared_ptr<batch_normalization_backward::primitive_desc> bwd_pd;
  shared_ptr<batch_normalization_backward> instancenorm_bwd;

  INBwdContext()
      : in_diff_mem(nullptr), src_mem(nullptr), gamma_mem(nullptr),
        beta_mem(nullptr), mean_mem(nullptr), var_mem(nullptr),
        weights_mem(nullptr), gamma_diff_mem(nullptr), out_diff_mem(nullptr),
        bwd_desc(nullptr), bwd_pd(nullptr), instancenorm_bwd(nullptr) {}

  ~INBwdContext() {
    in_diff_mem.reset();
    src_mem.reset();
    gamma_mem.reset();
    beta_mem.reset();
    mean_mem.reset();
    var_mem.reset();
    weights_mem.reset();
    gamma_diff_mem.reset();
    out_diff_mem.reset();
    bwd_desc.reset();
    bwd_pd.reset();
    instancenorm_bwd.reset();
  }

  size_t get_memory_used() {
    size_t memory_used_bytes = 0;
    if (in_diff_mem != nullptr)
      memory_used_bytes += in_diff_mem->get_desc().get_size();
    if (src_mem != nullptr)
      memory_used_bytes += src_mem->get_desc().get_size();
    if (gamma_mem != nullptr)
      memory_used_bytes += gamma_mem->get_desc().get_size();
    if (beta_mem != nullptr)
      memory_used_bytes += beta_mem->get_desc().get_size();
    if (mean_mem != nullptr)
      memory_used_bytes += mean_mem->get_desc().get_size();
    if (var_mem != nullptr)
      memory_used_bytes += var_mem->get_desc().get_size();
    if (weights_mem != nullptr)
      memory_used_bytes += weights_mem->get_desc().get_size();
    if (gamma_diff_mem != nullptr)
      memory_used_bytes += gamma_diff_mem->get_desc().get_size();
    if (out_diff_mem != nullptr)
      memory_used_bytes += out_diff_mem->get_desc().get_size();
    return memory_used_bytes;
  }
};

class InstanceNormalization : public Operator {
public:
  InstanceNormalization(string n, vector<string> i, vector<string> o, float e,
                        unordered_map<string, shared_ptr<Tensor>> &tensors,
                        int training)
      : Operator(n, "InstanceNormalization", i, o, tensors, training) {
    epsilon = e;
    auto b_i = vector<string>{"diff_" + o.at(0)};
    for (auto inp : i) {
      b_i.push_back(inp);
    }
    _f_op = ExecutionOp("fwd_" + n, "fwd", i, o);
    _b_op = ExecutionOp(
        "bwd_" + n, "bwd", b_i,
        vector<string>{"diff_" + i.at(0), "diff_" + i.at(1) + "_" + i.at(2)});
    _fwd_context = nullptr;
    _bwd_context = nullptr;
    init(tensors);
  }
  ~InstanceNormalization() {
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
    auto gamma_name = _f_op.input.at(1);
    auto beta_name = _f_op.input.at(2);
    auto out_name = _f_op.output.at(0);
    string mean_name, var_name;
    if (_f_op.input.size() > 3) {
      mean_name = _f_op.input.at(3);
    } else {
      mean_name = name + "_mean";
    }
    if (_f_op.input.size() > 4) {
      var_name = _f_op.input.at(4);
    } else {
      var_name = name + "_var";
    }
    auto src_md = tensors[src_name]->desc();
    auto src_dims = src_md.dims();
    auto batchsize = src_dims.at(0);
    auto channels = src_dims.at(1);
    auto time_name = getForwardTimeName(eng);
    auto s = dev->get_stream(0);
    if (_fwd_context == nullptr) {
      auto time_create = get_time();
      _fwd_context.reset(new INFwdContext());
      // get memory
      _fwd_context->src_mem.reset(new memory(tensors[src_name]->desc(), eng));
      _fwd_context->gamma_mem.reset(
          new memory(tensors[gamma_name]->desc(), eng));
      _fwd_context->beta_mem.reset(new memory(tensors[beta_name]->desc(), eng));
      _fwd_context->dst_mem.reset(new memory(tensors[src_name]->desc(), eng));
      tensors[out_name]->init(tensors[src_name]->desc(), dev);
      // maybe create statistic tensors
      auto stat_dims = memory::dims({batchsize, channels});
      auto stat_md =
          memory::desc(stat_dims, g_data_type, memory::format_tag::nc);
      if (_f_op.input.size() < 4) {
        tensors[mean_name] = move(make_shared<Tensor>(mean_name, stat_dims));
        tensors[mean_name]->init(stat_md, dev);
      }
      if (_f_op.input.size() < 5) {
        tensors[var_name] = move(make_shared<Tensor>(var_name, stat_dims));
        tensors[var_name]->init(stat_md, dev);
      }
      if (_fwd_context->mean_mem == nullptr) {
        _fwd_context->mean_mem.reset(
            new memory(tensors[mean_name]->desc(), eng));
      }
      if (_fwd_context->var_mem == nullptr) {
        _fwd_context->var_mem.reset(new memory(tensors[var_name]->desc(), eng));
      }
      timings[time_name][gamma_name] =
          maybe_do_reorder(tensors[gamma_name]->get_memory(),
                           *_fwd_context->gamma_mem, s, measure_time);
      timings[time_name][beta_name] =
          maybe_do_reorder(tensors[beta_name]->get_memory(),
                           *_fwd_context->beta_mem, s, measure_time);
      timings[time_name][mean_name] =
          maybe_do_reorder(tensors[mean_name]->get_memory(),
                           *_fwd_context->mean_mem, s, measure_time);
      timings[time_name][var_name] =
          maybe_do_reorder(tensors[var_name]->get_memory(),
                           *_fwd_context->var_mem, s, measure_time);
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
    auto time_exe = get_time();
    auto weights = vector<float>(2 * channels);
    auto gamma =
        static_cast<float *>(_fwd_context->gamma_mem->get_data_handle());
    auto beta = static_cast<float *>(_fwd_context->beta_mem->get_data_handle());
    copy_n(gamma, channels, weights.data());
    copy_n(beta, channels, weights.data() + channels);

    if (batchsize > 1) {
      // reorder input to standard memory format first
      memory::format_tag src_tag = get_ncdhw_tag(src_dims);

      auto src_mem_ncdhw =
          make_memory(memory::desc(src_dims, g_data_type, src_tag), eng);
      auto dst_mem_ncdhw =
          make_memory(memory::desc(src_dims, g_data_type, src_tag), eng);

      timings[time_name][src_name] = maybe_do_reorder(
          tensors[src_name]->get_memory(), src_mem_ncdhw, s, measure_time);
      auto new_dims = src_dims;
      new_dims.at(0) = 1;
      auto src_bn_md = memory::desc(new_dims, g_data_type, src_tag);
      auto stat_bn_md = memory::desc(memory::dims({1, channels}), g_data_type,
                                     memory::format_tag::nc);

      if (_fwd_context->fwd_desc == nullptr) {
        auto time_create = get_time();
        _fwd_context->fwd_desc.reset(new batch_normalization_forward::desc(
            {getMode(training), src_bn_md, epsilon,
             normalization_flags::use_scale_shift}));
        _fwd_context->fwd_pd.reset(
            new batch_normalization_forward::primitive_desc(
                *_fwd_context->fwd_desc, eng));
        _fwd_context->instancenorm_fwd.reset(
            new batch_normalization_forward(*_fwd_context->fwd_pd));
        timings[time_name]["create"] += get_elapsed_ms(time_create);
        if (_track_only_tensor_memory == 0) {
          dev->memory_used += _fwd_context->get_memory_used();
        }
      }
      auto const weights_mem = make_memory(
          _fwd_context->fwd_pd.get()->weights_desc(), eng, weights.data());
      auto src_data = static_cast<float *>(src_mem_ncdhw.get_data_handle());
      auto mean_data =
          static_cast<float *>(_fwd_context->mean_mem->get_data_handle());
      auto var_data =
          static_cast<float *>(_fwd_context->var_mem->get_data_handle());
      auto dst_data = static_cast<float *>(dst_mem_ncdhw.get_data_handle());
      size_t volume = 1;
      for (auto i = 1; i < src_dims.size(); i++) {
        volume *= src_dims.at(i);
      }
      for (auto n = 0; n < batchsize; n++) {
        auto const startSample = n * volume;
        auto const startStatistics = n * channels;
        auto src_bn_vec = vector<float>(volume);
        auto mean_bn_vec = vector<float>(channels);
        auto var_bn_vec = vector<float>(channels);
        copy_n(src_data + startSample, volume, src_bn_vec.data());
        copy_n(mean_data + startStatistics, channels, mean_bn_vec.data());
        copy_n(var_data + startStatistics, channels, var_bn_vec.data());
        auto src_bn_mem = make_memory(_fwd_context->fwd_pd.get()->src_desc(),
                                      eng, src_bn_vec.data());
        auto mean_bn_mem = make_memory(stat_bn_md, eng, mean_bn_vec.data());
        auto var_bn_mem = make_memory(stat_bn_md, eng, var_bn_vec.data());
        auto dst_bn_mem =
            make_memory(_fwd_context->fwd_pd.get()->dst_desc(), eng);
        auto args = unordered_map<int, memory>({{DNNL_ARG_SRC, src_bn_mem},
                                                {DNNL_ARG_MEAN, mean_bn_mem},
                                                {DNNL_ARG_VARIANCE, var_bn_mem},
                                                {DNNL_ARG_WEIGHTS, weights_mem},
                                                {DNNL_ARG_DST, dst_bn_mem}});
        _fwd_context->instancenorm_fwd->execute(s, args);
        s.wait();
        auto dst_bn_data = static_cast<float *>(dst_bn_mem.get_data_handle());
        copy_n(dst_bn_data, volume, dst_data + startSample);
      }
    } else {
      if (_fwd_context->fwd_desc == nullptr) {
        auto time_create = get_time();
        _fwd_context->fwd_desc.reset(new batch_normalization_forward::desc(
            {getMode(training), src_md, epsilon,
             normalization_flags::use_scale_shift}));
        _fwd_context->fwd_pd.reset(
            new batch_normalization_forward::primitive_desc(
                *_fwd_context->fwd_desc, eng));
        _fwd_context->instancenorm_fwd.reset(
            new batch_normalization_forward(*_fwd_context->fwd_pd));
        timings[time_name]["create"] += get_elapsed_ms(time_create);
        if (_track_only_tensor_memory == 0) {
          dev->memory_used += _fwd_context->get_memory_used();
        }
      }
      auto const weights_mem = make_memory(
          _fwd_context->fwd_pd.get()->weights_desc(), eng, weights.data());
      auto args = unordered_map<int, memory>(
          {{DNNL_ARG_SRC, *_fwd_context->src_mem},
           {DNNL_ARG_MEAN, *_fwd_context->mean_mem},
           {DNNL_ARG_VARIANCE, *_fwd_context->var_mem},
           {DNNL_ARG_WEIGHTS, weights_mem},
           {DNNL_ARG_DST, *_fwd_context->dst_mem}});
      _fwd_context->instancenorm_fwd->execute(s, args);
      s.wait();
    }
    if (measure_time) {
      timings[time_name]["exe"] = get_elapsed_ms(time_exe);
      timings[time_name]["total"] = get_elapsed_ms(begin);
    }
  }

  void backward(shared_ptr<Device> dev,
                unordered_map<string, shared_ptr<Tensor>> &tensors,
                memory::format_tag outputTag, const int measure_time) {
    _b_device = dev;
    throw runtime_error("InstanceNormalization is not yet implemented!");
  }
  float epsilon;
  shared_ptr<INFwdContext> _fwd_context;
  shared_ptr<INBwdContext> _bwd_context;
};
#endif
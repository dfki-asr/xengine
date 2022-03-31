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

  size_t get_memory_used() {
    size_t memory_used_bytes = 0;
    for (auto src : src_mem) {
      if (src != nullptr)
        memory_used_bytes += src->get_desc().get_size();
    }
    if (dst_mem != nullptr)
      memory_used_bytes += dst_mem->get_desc().get_size();
    return memory_used_bytes;
  }
};

inline void concat_bwd(dnnl::memory &in_mem, dnnl::memory &out_mem,
                       const size_t offset) {
  dnnl::engine dst_eng = out_mem.get_engine();
  dnnl::engine src_eng = in_mem.get_engine();
  assert(g_data_type == memory::data_type::f32);
  size_t size = out_mem.get_desc().get_size() / sizeof(g_data_type);
#ifdef DNNL_WITH_SYCL
  bool is_cpu_sycl = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL &&
                      dst_eng.get_kind() == dnnl::engine::kind::cpu);
  bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL &&
                      dst_eng.get_kind() == dnnl::engine::kind::gpu);
  if (is_cpu_sycl || is_gpu_sycl) {
    auto mkind = dnnl::sycl_interop::get_memory_kind(out_mem);
    if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
      auto dst_buffer = dnnl::sycl_interop::get_buffer<float>(out_mem);
      auto dst = dst_buffer.get_access<::sycl::access::mode::write>();
      float *dst_ptr = dst.get_pointer();
      if (!dst_ptr)
        throw std::runtime_error("get_pointer returned nullptr.");
      auto src_buffer = dnnl::sycl_interop::get_buffer<float>(in_mem);
      auto src = src_buffer.get_access<::sycl::access::mode::read>();
      float *src_ptr = src.get_pointer();
      if (!src_ptr)
        throw std::runtime_error("get_pointer returned nullptr.");
      for (size_t i = 0; i < size; ++i) {
        dst_ptr[i] = src_ptr[offset + i];
      }
    } else {
      assert(mkind == dnnl::sycl_interop::memory_kind::usm);
      float *dst_ptr = static_cast<float *>(out_mem.get_data_handle());
      if (!dst_ptr)
        throw std::runtime_error("get_data_handle returned nullptr.");
      float *src_ptr = static_cast<float *>(in_mem.get_data_handle());
      if (!src_ptr)
        throw std::runtime_error("get_data_handle returned nullptr.");
      if (is_cpu_sycl) {
        for (size_t i = 0; i < size; ++i) {
          dst_ptr[i] = src_ptr[offset + i];
        }
      } else {
        auto sycl_queue = dnnl::sycl_interop::get_queue(dnnl::stream(dst_eng));
        auto execute_queue = sycl_queue.submit([&](::sycl::handler &cgh) {
          cgh.parallel_for<kernel_tag>(::sycl::range<1>(size),
                                       [=](::sycl::id<1> i) {
                                         int idx = (int)i[0];
                                         dst_ptr[idx] = src_ptr[offset + i];
                                       });
        });
      }
    }
    return;
  }
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
  if (dst_eng.get_kind() == dnnl::engine::kind::gpu ||
      src_eng.get_kind() == dnnl::engine::kind::gpu) {
    void *mapped_dst_ptr = out_mem.map_data();
    void *mapped_src_ptr = in_mem.map_data();
    if (mapped_dst_ptr && mapped_src_ptr) {
      for (size_t i = 0; i < size; ++i) {
        mapped_dst_ptr[i] = mapped_src_ptr[offset + i];
      }
    }
    out_mem.unmap_data(mapped_dst_ptr);
    return;
  }
#endif

  if (dst_eng.get_kind() == dnnl::engine::kind::cpu &&
      src_eng.get_kind() == dnnl::engine::kind::cpu) {
    float *dst = static_cast<float *>(out_mem.get_data_handle());
    if (!dst)
      throw std::runtime_error("get_data_handle returned nullptr.");
    float *src = static_cast<float *>(in_mem.get_data_handle());
    if (!src)
      throw std::runtime_error("get_data_handle returned nullptr.");
    for (size_t i = 0; i < size; ++i) {
      dst[i] = src[offset + i];
    }
    return;
  }

  assert(!"not expected");
}

class Concat : public Operator {
public:
  Concat(string n, vector<string> i, vector<string> o, int a,
         unordered_map<string, shared_ptr<Tensor>> &tensors, int training)
      : Operator(n, "Concat", i, o, tensors, training) {
    axis = a;
    _f_op = ExecutionOp("fwd_" + n, "fwd", i, o);
    auto b_o = vector<string>();
    for (size_t j = 0; j < i.size(); j++) {
      b_o.push_back("diff_" + i.at(j));
    }
    _b_op =
        ExecutionOp("bwd_" + n, "bwd", vector<string>{"diff_" + o.at(0)}, b_o);
    _fwd_context = nullptr;
    init(tensors);
  }
  ~Concat() { reset_fwd_primitives(); }
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
    auto out_name = _f_op.output.at(0);
    auto time_name = getForwardTimeName(dev->name);
    if (_fwd_context == nullptr) {
      auto time_create = get_time();
      _fwd_context.reset(new ConcatFwdContext());
      timings[time_name]["create"] = get_elapsed_ms(time_create);
    }
    auto s = dev->get_stream(0);
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
      tensors[out_name]->init(_fwd_context->fwd_pd.get()->dst_desc(), dev);
      timings[time_name]["create"] += get_elapsed_ms(time_create);
      if (_track_only_tensor_memory == 0) {
        dev->memory_used += _fwd_context->get_memory_used();
      }
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

  void backward(shared_ptr<Device> dev,
                unordered_map<string, shared_ptr<Tensor>> &tensors,
                memory::format_tag outputTag, const int measure_time) {
    _b_device = dev;
    auto begin = get_time();
    auto eng = dev->get_engine();
    auto time_name = getBackwardTimeName(dev->name);
    auto in_diff_name = _b_op.input.at(0);
    // get memory
    auto in_diff_mem = make_memory(tensors[in_diff_name]->desc(), eng);
    auto s = dev->get_stream(0);
    timings[time_name][in_diff_name] = maybe_do_reorder(
        tensors[in_diff_name]->get_memory(), in_diff_mem, s, measure_time);
    size_t offset = 0;
    for (size_t i = 0; i < _b_op.output.size(); ++i) {
      auto out_diff_name = _b_op.output.at(i);
      auto src_name = _f_op.input.at(i);
      auto out_diff_desc = tensors[src_name]->desc();
      auto out_diff_mem = make_memory(out_diff_desc, eng);
      // copy correct part of in_diff_name to out_diff_mem
      concat_bwd(in_diff_mem, out_diff_mem, offset);
      tensors[out_diff_name]->init(out_diff_desc, dev);
      tensors[out_diff_name]->set_memory(out_diff_mem);
      offset += product(tensors[src_name]->dims());
    }
    timings[time_name]["create"] = 0.0f;
    if (measure_time) {
      timings[time_name]["exe"] = get_elapsed_ms(begin);
      timings[time_name]["total"] = get_elapsed_ms(begin);
    }
  }
  int axis;
  shared_ptr<ConcatFwdContext> _fwd_context;
};
#endif
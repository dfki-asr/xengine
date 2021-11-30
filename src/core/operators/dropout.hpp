#ifndef XENGINE_OP_DROPOUT_HPP
#define XENGINE_OP_DROPOUT_HPP

#include "../operator.hpp"

class kernel_tag;

inline void dropout(dnnl::memory &in_mem, dnnl::memory &out_mem,
                    const std::vector<uint8_t> &v) {
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
        dst_ptr[i] = v.at(i) > 0 ? src_ptr[i] : 0;
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
          dst_ptr[i] = v.at(i) > 0 ? src_ptr[i] : 0;
        }
      } else {
        auto sycl_queue = dnnl::sycl_interop::get_queue(dnnl::stream(dst_eng));
        auto execute_queue = sycl_queue.submit([&](::sycl::handler &cgh) {
          cgh.parallel_for<kernel_tag>(
              ::sycl::range<1>(size), [=](::sycl::id<1> i) {
                int idx = (int)i[0];
                dst_ptr[idx] = v.at(i) > 0 ? src_ptr[i] : 0;
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
        mapped_dst_ptr[i] = v.at(i) > 0 ? mapped_src_ptr[i] : 0;
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
      dst[i] = v.at(i) > 0 ? src[i] : 0;
    }
    return;
  }

  assert(!"not expected");
}

class Dropout : public Operator {
public:
  Dropout(string n, vector<string> i, vector<string> o, float p,
          unordered_map<string, unique_ptr<Tensor>> &tensors, int training)
      : Operator(n, "Dropout", i, o, tensors, training) {
    probability = p;
    _f_op = ExecutionOp("fwd_" + n, "fwd", i, o);
    _b_op = ExecutionOp("bwd_" + n, "bwd", vector<string>{"diff_" + o.at(0)},
                        vector<string>{"diff_" + i.at(0)});
    init(tensors);
  }

  void forward(Device &dev, unordered_map<string, unique_ptr<Tensor>> &tensors,
               memory::format_tag outputTag, const int measure_time) {
    auto begin = get_time();
    auto eng = dev.get_engine();
    auto time_exe = get_time();
    auto src_name = _f_op.input.at(0);
    auto out_name = _f_op.output.at(0);
    auto time_name = getForwardTimeName(eng);
    if (training) {
      auto src_md = tensors[src_name]->desc();
      // get memory
      auto src_mem = make_memory(src_md, eng);
      tensors[out_name]->init(src_md, eng);
      auto dst_mem = tensors[out_name]->get_memory();
      // reorders
      auto s = stream(eng);
      timings[time_name][src_name] = maybe_do_reorder(
          tensors[src_name]->get_memory(), src_mem, s, measure_time);
      // execute
      size_t size = src_md.get_size() / sizeof(float);
      mask = std::vector<uint8_t>(size);
      srand(time(0));
      generate(mask.begin(), mask.end(), _rand_uint8_t);
      uint8_t P = probability * 100;
      for (size_t i = 0; i < size; ++i) {
        mask[i] = mask.at(i) > P ? 1 : 0;
      }
      time_exe = get_time();
      dropout(src_mem, dst_mem, mask);
    } else {
      throw runtime_error("Dropout->forward in inference mode not supported!");
    }
    timings[time_name]["create"] = 0.0f;
    if (measure_time) {
      timings[time_name]["exe"] = get_elapsed_ms(time_exe);
      timings[time_name]["total"] = get_elapsed_ms(begin);
    }
  }

  void backward(Device &dev, unordered_map<string, unique_ptr<Tensor>> &tensors,
                memory::format_tag outputTag, const int measure_time) {
    auto begin = get_time();
    auto eng = dev.get_engine();
    auto in_diff_name = _b_op.input.at(0);
    auto out_diff_name = _b_op.output.at(0);
    auto src_md = tensors[_f_op.input.at(0)]->desc();
    assert(tensors.find(in_diff_name) != tensors.end());
    auto time_name = getBackwardTimeName(eng);
    // get memory
    auto in_diff_mem = make_memory(src_md, eng);
    tensors[out_diff_name]->init(src_md, eng);
    auto dst_mem = tensors[out_diff_name]->get_memory();
    // reorders
    auto s = stream(eng);
    timings[time_name][in_diff_name] = maybe_do_reorder(
        tensors[in_diff_name]->get_memory(), in_diff_mem, s, measure_time);
    auto time_exe = get_time();
    // execute
    dropout(in_diff_mem, dst_mem, mask);
    timings[time_name]["create"] = 0.0f;
    if (measure_time) {
      timings[time_name]["exe"] = get_elapsed_ms(time_exe);
      timings[time_name]["total"] = get_elapsed_ms(begin);
    }
  }

  float probability;
  vector<uint8_t> mask;
};
#endif
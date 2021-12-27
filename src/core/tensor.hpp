#ifndef XENGINE_TENSOR_HPP
#define XENGINE_TENSOR_HPP

#include "../xengine_utils.hpp"

using namespace std;
using namespace dnnl;

class Tensor {
public:
  Tensor(string n, memory::dims dims) : name(n), _dims(dims) {
    _mem = nullptr;
    _desc = memory::desc();
    _eng = dnnl::engine();
    producer = "";
    consumers = vector<string>();
    _size_in_bytes = 0;
  }

  Tensor(string n, memory::desc d, engine &e) : name(n) {
    init(d, e);
    producer = "";
    consumers = vector<string>();
  }

  void add_consumer(const string consumer) {
    if (find(consumers.begin(), consumers.end(), consumer) == consumers.end()) {
      consumers.push_back(consumer);
    }
  }

  void reinit(memory::desc d) { init(d, _eng); }

  void init(memory::desc d, engine &e) {
    if (_mem != nullptr) {
      release();
    }
    _mem = make_unique<memory>(move(make_memory(d, e)));
    _desc = d;
    _eng = _mem->get_engine();
    _dims = d.dims();
    _size_in_bytes = d.get_size();
  }

  memory::dims dims() {
    return (_mem != nullptr) ? _mem->get_desc().dims() : _dims;
  }

  int size() { return _size_in_bytes; }

  int is_initialized() { return _mem != nullptr; }

  void ensure_initialization() {
    if (!is_initialized())
      throw runtime_error(
          name + "'s memory is not initialized! (Producer: " + producer + ")");
  }

  memory::desc desc() {
    if (!is_initialized()) {
      if (_desc == memory::desc()) {
        throw runtime_error(name + " has no valid descriptor!");
      }
      return _desc;
    }
    return _mem->get_desc();
  }

  memory &get_memory() {
    ensure_initialization();
    return *_mem;
  }

  engine get_engine() {
    if (!is_initialized()) {
      if (_eng == dnnl::engine()) {
        throw runtime_error(name + " has no valid engine!");
      }
      return _eng;
    }
    return _mem->get_engine();
  }

  void set_memory(const memory &mem) {
    if (is_initialized()) {
      release();
    }
    dnnl::engine eng = mem.get_engine();
    size_t size = mem.get_desc().get_size() / sizeof(g_data_type);
    _mem = make_unique<memory>(move(make_memory(mem.get_desc(), eng)));
    _eng = _mem->get_engine();
    _size_in_bytes = mem.get_desc().get_size();
#ifdef DNNL_WITH_SYCL
    bool is_cpu_sycl = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL &&
                        _eng.get_kind() == dnnl::engine::kind::cpu);
    bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL &&
                        _eng.get_kind() == dnnl::engine::kind::gpu);
    if (is_cpu_sycl || is_gpu_sycl) {
      auto mkind = dnnl::sycl_interop::get_memory_kind(*_mem);
      if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
        auto dst_buffer = dnnl::sycl_interop::get_buffer<float>(*_mem);
        auto dst = dst_buffer.get_access<::sycl::access::mode::write>();
        float *dst_ptr = dst.get_pointer();
        if (!dst_ptr)
          throw std::runtime_error("get_pointer returned nullptr.");
        auto src_buffer = dnnl::sycl_interop::get_buffer<float>(mem);
        auto src = src_buffer.get_access<::sycl::access::mode::read>();
        float *src_ptr = src.get_pointer();
        if (!src_ptr)
          throw std::runtime_error("get_pointer returned nullptr.");
        for (size_t i = 0; i < size; ++i)
          dst_ptr[i] = src_ptr[i];
      } else {
        assert(mkind == dnnl::sycl_interop::memory_kind::usm);
        float *dst_ptr = static_cast<float *>(_mem->get_data_handle());
        if (!dst_ptr)
          throw std::runtime_error("get_data_handle returned nullptr.");
        float *src_ptr = static_cast<float *>(mem.get_data_handle());
        if (!src_ptr)
          throw std::runtime_error("get_data_handle returned nullptr.");
        if (is_cpu_sycl) {
          for (size_t i = 0; i < size; ++i) {
            dst_ptr[i] = src_ptr[i];
          }
        } else {
          auto sycl_queue = dnnl::sycl_interop::get_queue(dnnl::stream(_eng));
          sycl_queue.memcpy(dst_ptr, src_ptr, size).wait();
        }
      }
      return;
    }
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (_eng.get_kind() == dnnl::engine::kind::gpu ||
        eng.get_kind() == dnnl::engine::kind::gpu) {
      void *mapped_dst_ptr = _mem->map_data();
      void *mapped_src_ptr = mem.map_data();
      if (mapped_dst_ptr && mapped_src_ptr)
        std::memcpy(mapped_dst_ptr, mapped_src_ptr, size);
      _mem->unmap_data(mapped_dst_ptr);
      return;
    }
#endif

    if (_eng.get_kind() == dnnl::engine::kind::cpu &&
        eng.get_kind() == dnnl::engine::kind::cpu) {
      float *dst = static_cast<float *>(_mem->get_data_handle());
      if (!dst)
        throw std::runtime_error("get_data_handle returned nullptr.");
      float *src = static_cast<float *>(mem.get_data_handle());
      if (!src)
        throw std::runtime_error("get_data_handle returned nullptr.");
      for (size_t i = 0; i < size; ++i) {
        dst[i] = src[i];
      }
      return;
    }

    assert(!"not expected");
  }

  void set_dims(memory::dims d) {
    _dims = d;
    auto tag =
        d.size() == 4 ? memory::format_tag::oihw : memory::format_tag::oi;
    if (_mem != nullptr) {
      auto eng = _mem->get_engine();
      release();
      _mem = make_unique<memory>(
          move(make_memory(memory::desc(d, g_data_type, tag), eng)));
      _size_in_bytes = _mem->get_desc().get_size();
    }
  }

  void release() {
    if (_mem != nullptr) {
      _mem.reset();
      _mem.release();
      _mem = nullptr;
    }
  }

  string name;
  unique_ptr<memory> _mem;
  string producer;
  vector<string> consumers;

private:
  memory::dims _dims;
  memory::desc _desc;
  dnnl::engine _eng;
  int _size_in_bytes;
};
#endif
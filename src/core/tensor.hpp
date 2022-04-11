#ifndef XENGINE_TENSOR_HPP
#define XENGINE_TENSOR_HPP

#include "../xengine_utils.hpp"

using namespace std;
using namespace dnnl;

class Tensor {

public:
  Tensor(const string n, const memory::dims dims)
      : _name(n), _mem(nullptr), _dims(dims), _desc(memory::desc()),
        _descs(unordered_map<string, memory::desc>()), _eng(dnnl::engine()),
        _device(nullptr), _producer(""), _size(0),
        _consumers(vector<string>()) {}

  Tensor(const string n, const memory::desc d, engine &e)
      : _name(n), _mem(nullptr), _dims(d.dims()), _desc(d),
        _descs(unordered_map<string, memory::desc>()), _eng(e),
        _device(nullptr), _producer(""), _size(0),
        _consumers(vector<string>()) {
    init(d, e);
  }

  void init(const memory::desc d, shared_ptr<Device> dev) {
    if (is_initialized()) {
      release();
    }
    _device = dev;
    _descs[dev->name] = d;
    init(d, dev->get_engine());
  }

  void init(const memory::desc d, engine &e) {
    if (is_initialized()) {
      release();
    }
    _mem = make_unique<memory>(move(make_memory(d, e)));
    _dims = d.dims();
    _desc = d;
    _size = _mem->get_desc().get_size();
    _eng = _mem->get_engine();
    if (_device != nullptr) {
      _device->memory_used += _mem->get_desc().get_size();
    } else {
      throw runtime_error("device is nullptr for tensor " + _name);
    }
  }

  void reinit(const memory::desc d) { init(d, _eng); }

  void release() {
    if (is_initialized()) {
      if (_device != nullptr) {
        _device->memory_used -= _mem->get_desc().get_size();
      } else {
        throw runtime_error("device is nullptr for tensor " + _name);
      }
      _mem.reset();
      _mem.release();
      _mem = nullptr;
    }
  }

  void set_producer(const string producer) { _producer = producer; }

  void add_consumer(const string consumer) {
    if (find(_consumers.begin(), _consumers.end(), consumer) ==
        _consumers.end()) {
      _consumers.push_back(consumer);
    }
  }

  void set_dims(const memory::dims d) {
    _dims = d;
    auto tag =
        d.size() == 4 ? memory::format_tag::oihw : memory::format_tag::oi;
    if (is_initialized()) {
      auto eng = _mem->get_engine();
      release();
      _mem = make_unique<memory>(
          move(make_memory(memory::desc(d, g_data_type, tag), eng)));
    }
  }

  shared_ptr<Device> get_device() { return _device; }

  void set_device(shared_ptr<Device> device) { _device = device; }

  void set_memory(const memory &mem) {
    if (is_initialized()) {
      release();
    }
    dnnl::engine eng = mem.get_engine();
    if (_device != nullptr) {
      _device->memory_used += mem.get_desc().get_size();
    } else {
      throw runtime_error("device is nullptr for tensor " + _name);
    }
    size_t size = mem.get_desc().get_size() / sizeof(g_data_type);
    _mem = make_unique<memory>(move(make_memory(mem.get_desc(), eng)));
    _dims = _mem->get_desc().dims();
    _desc = _mem->get_desc();
    _size = _mem->get_desc().get_size();
    _eng = _mem->get_engine();
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

  int is_initialized() { return _mem != nullptr; }

  string name() { return _name; }

  memory::dims dims() {
    return (is_initialized()) ? _mem->get_desc().dims() : _dims;
  }

  memory::desc desc() {
    if (!is_initialized()) {
      if (_desc == memory::desc()) {
        throw runtime_error(_name + " has no valid descriptor!");
      }
      return _desc;
    }
    return _mem->get_desc();
  }

  memory::desc desc(string dev_name) {
    if (_descs.find(dev_name) == _descs.end()) {
      throw runtime_error(_name + " has no valid descriptor for device " +
                          dev_name + "!");
    }
    return _descs[dev_name];
  }

  long long get_size() {
    if (_mem != nullptr) {
      return _mem->get_desc().get_size();
    }
    return _size;
  }

  engine get_engine() {
    if (!is_initialized()) {
      if (_eng == dnnl::engine()) {
        throw runtime_error(_name + " has no valid engine!");
      }
      return _eng;
    }
    return _mem->get_engine();
  }

  memory &get_memory() {
    if (!is_initialized())
      throw runtime_error(_name + "'s memory is not initialized! (Producer: " +
                          _producer + ")");
    return *_mem;
  }

  string producer() { return _producer; }

  vector<string> consumers() { return _consumers; }

private:
  string _name;
  unique_ptr<memory> _mem;
  memory::dims _dims;
  memory::desc _desc;
  unordered_map<string, memory::desc> _descs;
  long long _size;
  dnnl::engine _eng;
  shared_ptr<Device> _device;
  string _producer;
  vector<string> _consumers;
};
#endif
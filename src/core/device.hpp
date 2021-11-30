#ifndef XENGINE_DEVICE_HPP
#define XENGINE_DEVICE_HPP

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "dnnl_sycl.hpp"
#endif
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OCL
#include "dnnl_ocl.hpp"
#endif

using namespace std;
using namespace dnnl;

class Device {
public:
  Device(vector<string> &i)
      : Device(i.at(0), i.at(1), stoi(i.at(2)), stoi(i.at(3)), stoul(i.at(4))) {
  }
  Device(string n, string type, int devID, int num_streams, size_t b = 0)
      : name(n), eng(nullptr) {
    if (num_streams < 0) {
      throw runtime_error(
          "Cannot create Device instance due to negative number of streams.");
    }
    engine::kind engine_kind;
    if (type == "cpu") {
      engine_kind = engine::kind::cpu;
    } else if (type == "gpu") {
      engine_kind = engine::kind::gpu;
    } else {
      throw runtime_error("Invalid device type.");
    }
    if (eng == nullptr) {
      if (type == "cpu") {
        auto dev = sycl_interop::get_device(engine(engine_kind, devID));
        auto max_sub_dev =
            dev.get_info<cl::sycl::info::device::partition_max_sub_devices>();
        auto nproc = dev.get_info<cl::sycl::info::device::max_compute_units>();
        if (max_sub_dev > 0) {
          const char *num_cores = getenv("OPENCL_NUM_CORES");
          if (num_cores != nullptr) {
            size_t num_cores_dev0 = stoi(num_cores);
            if (num_cores_dev0 < nproc) {
              cout << "use " << num_cores_dev0 << "/" << nproc
                   << " compute units." << endl;
              auto sub_dev = dev.create_sub_devices<
                  cl::sycl::info::partition_property::partition_by_counts>(
                  {num_cores_dev0, nproc - num_cores_dev0});
              const auto d0 = sub_dev[0];
              if (d0.get_info<cl::sycl::info::device::max_compute_units>() !=
                  num_cores_dev0) {
                throw runtime_error(
                    "Subdevice 0 was not successfully created.");
              }
              const cl::sycl::context sub_ctx0(d0);
              eng =
                  make_unique<engine>(sycl_interop::make_engine(d0, sub_ctx0));
            }
          }
        }
      }
    }
    if (eng == nullptr) {
      eng = make_unique<engine>(engine_kind, devID);
    }
    auto dev = sycl_interop::get_device(*eng.get());
    cout << "Created device of type " << type << " with "
         << dev.get_info<cl::sycl::info::device::max_compute_units>()
         << " compute units." << endl;
    for (auto i = 0; i < num_streams; i++) {
      streams.push_back(make_unique<stream>(*eng, stream::flags::in_order));
    }
    budget = (b == 0) ? global_mem_size_bytes() : b;
  }
  ~Device() {
    if (eng != nullptr) {
      eng.reset();
      eng.release();
      eng = nullptr;
    }
    for (auto i = 0; i < streams.size(); i++) {
      streams.at(i).reset();
      streams.at(i).release();
    }
  }

  size_t global_mem_size_bytes() {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
    return sycl_interop::get_device(*eng.get())
        .get_info<cl::sycl::info::device::global_mem_size>();
#endif
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OCL
    return ocl_interop::get_device(*eng.get())
        .get_info<cl_device::global_mem_size>();
#endif
  }

  size_t max_mem_alloc_size_bytes() {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
    return sycl_interop::get_device(*eng.get())
        .get_info<cl::sycl::info::device::max_mem_alloc_size>();
#endif
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OCL
    return ocl_interop::get_device(*eng.get())
        .get_info<cl_device::max_mem_alloc_size>();
#endif
  }

  size_t local_mem_size_bytes() {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
    return sycl_interop::get_device(*eng.get())
        .get_info<cl::sycl::info::device::local_mem_size>();
#endif
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OCL
    return ocl_interop::get_device(*eng.get())
        .get_info<cl_device::local_mem_size>();
#endif
  }

  engine &get_engine() { return *eng; }

  stream &get_stream(const int idx) { return *streams.at(idx); }

  string name;
  unique_ptr<engine> eng;
  vector<unique_ptr<stream>> streams;
  float budget;
};
#endif
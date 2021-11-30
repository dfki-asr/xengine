#include "../src/core/device.hpp"
#include "../src/core/tensor.hpp"
#include "../src/onnx_utils.hpp"
#include "dnnl.hpp"
#include <vector>

using namespace dnnl;
using namespace std;

vector<Tensor> create_tensors(map<string, shared_ptr<Device>> &devices) {
  auto tensors = vector<Tensor>();
  auto cpu_engine = devices["cpu_0"]->get_engine();
  auto gpu_engine = devices["gpu_0"]->get_engine();
  auto data_type = memory::data_type::f32;
  auto format_tag = memory::format_tag::nchw;
  memory::dims pool1_src_tz = {64, 20, 24, 24};
  memory::dims pool1_dst_tz = {64, 20, 12, 12};
  tensors.push_back(Tensor("pool1_src_cpu",
                           memory::desc({pool1_src_tz, data_type, format_tag}),
                           cpu_engine));
  tensors.push_back(Tensor("pool1_src_gpu",
                           memory::desc({pool1_src_tz, data_type, format_tag}),
                           gpu_engine));
  tensors.push_back(Tensor("pool1_dst_cpu",
                           memory::desc({pool1_dst_tz, data_type, format_tag}),
                           cpu_engine));
  tensors.push_back(Tensor("pool1_dst_gpu",
                           memory::desc({pool1_dst_tz, data_type, format_tag}),
                           gpu_engine));
  return tensors;
}

void create_primitives(map<string, shared_ptr<Device>> &devices,
                       vector<Tensor> &memory_vec,
                       vector<primitive> &primitives,
                       vector<unordered_map<int, memory>> &primitive_args) {
  auto cpu_engine = devices["cpu_0"]->get_engine();
  auto gpu_engine = devices["gpu_0"]->get_engine();
  auto pool1_m_src_cpu = memory_vec.at(0).get_memory();
  auto pool1_m_src_gpu = memory_vec.at(1).get_memory();
  auto pool1_m_dst_cpu = memory_vec.at(2).get_memory();
  auto pool1_m_dst_gpu = memory_vec.at(3).get_memory();
  memory::dims pool1_kernel = {2, 2};
  memory::dims pool1_strides = {2, 2};
  memory::dims pool1_padding = {0, 0};

  // // Reorder CPU -> GPU
  auto r1 = reorder(pool1_m_src_cpu, pool1_m_src_gpu);
  primitives.push_back(r1);
  primitive_args.push_back(
      {{DNNL_ARG_FROM, pool1_m_src_cpu}, {DNNL_ARG_TO, pool1_m_src_gpu}});

  // Relu
  auto relu_desc =
      eltwise_forward::desc(prop_kind::forward, algorithm::eltwise_relu,
                            pool1_m_src_gpu.get_desc(), 0.0f);
  auto relu_pd = eltwise_forward::primitive_desc(relu_desc, gpu_engine);
  auto relu = eltwise_forward(relu_pd);
  primitives.push_back(relu);
  primitive_args.push_back(
      {{DNNL_ARG_SRC, pool1_m_src_gpu}, {DNNL_ARG_DST, pool1_m_src_gpu}});

  // Pool
  auto pool1_desc = pooling_forward::desc(
      prop_kind::forward_inference, algorithm::pooling_max,
      pool1_m_src_gpu.get_desc(), pool1_m_dst_gpu.get_desc(), pool1_strides,
      pool1_kernel, pool1_padding, pool1_padding);
  auto pool1_pd = pooling_forward::primitive_desc(pool1_desc, gpu_engine);
  auto pool1 = pooling_forward(pool1_pd);
  primitives.push_back(pool1);
  primitive_args.push_back(
      {{DNNL_ARG_SRC, pool1_m_src_gpu}, {DNNL_ARG_DST, pool1_m_dst_gpu}});

  /// reorder GPU -> CPU
  auto r2 = reorder(pool1_m_dst_gpu, pool1_m_dst_cpu);
  primitives.push_back(r2);
  primitive_args.push_back(
      {{DNNL_ARG_FROM, pool1_m_dst_gpu}, {DNNL_ARG_TO, pool1_m_dst_cpu}});
}

void execute(map<string, shared_ptr<Device>> &devices,
             vector<primitive> &primitives,
             vector<unordered_map<int, memory>> &primitive_args) {
  auto s = devices["gpu_0"]->get_stream(0);
  for (uint8_t i = 0; i < primitives.size(); i++) {
    primitives.at(i).execute(s, primitive_args.at(i));
  }
  s.wait();
}

int main() {
  auto begin = get_time();
  auto devices = map<string, shared_ptr<Device>>();
  create_devices(devices, "../data/models/devices.txt");
  std::cout << "Device creation time: " << (get_elapsed_ms(begin)) << " ms."
            << std::endl;

  begin = get_time();
  vector<Tensor> tensors = create_tensors(devices);
  std::cout << "Memory creation time: " << (get_elapsed_ms(begin)) << " ms."
            << std::endl;

  begin = get_time();
  vector<primitive> primitives;
  vector<unordered_map<int, memory>> primitive_args;
  create_primitives(devices, tensors, primitives, primitive_args);
  std::cout << "Create Primitives time: " << (get_elapsed_ms(begin)) << " ms."
            << std::endl;

  begin = get_time();
  execute(devices, primitives, primitive_args);
  std::cout << "Execution time: " << (get_elapsed_ms(begin)) << " ms."
            << std::endl;
  return 1;
}

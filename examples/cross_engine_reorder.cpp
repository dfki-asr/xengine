#include "../src/example_utils.hpp"
#include "../src/xengine_utils.hpp"
#include "dnnl.hpp"
#include <chrono>
#include <vector>

using namespace dnnl;
using namespace std;

/// Three creation steps:
/// 1. Initialize an operation descriptor
/// 2. Create operation primitive descriptor on GPU engine
/// 3. Create a primitive with GPU memory objects to compute on GPU
///    Primitive creation might be a very expensive operation, so consider
///    creating primitive objects once and executing them multiple times.

void cross_engine_reorder() {
  auto begin = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
                   .count();
  auto cpu_engine = engine(engine::kind::cpu, 0);
  auto gpu_engine = engine(engine::kind::gpu, 0);
  auto stream_gpu = stream(gpu_engine, stream::flags::in_order);
  auto end = std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now().time_since_epoch())
                 .count();
  std::cout << "Engine setup time: " << (end - begin) << " ms." << std::endl;
  begin = end;

  /// Fill data in CPU memory and move from CPU to GPU by reorder
  auto batch = 64;
  memory::dims pool1_src_tz = {batch, 20, 24, 24};
  memory::dims pool1_dst_tz = {batch, 20, 12, 12};
  memory::dims pool1_kernel = {2, 2};
  memory::dims pool1_strides = {2, 2};
  memory::dims pool1_padding = {0, 0};
  // Create memory
  auto pool1_m_src_cpu = make_memory(
      {pool1_src_tz, memory::data_type::f32, memory::format_tag::nchw},
      cpu_engine);
  auto pool1_m_src_gpu = make_memory(
      {pool1_src_tz, memory::data_type::f32, memory::format_tag::nchw},
      gpu_engine);
  auto pool1_m_dst_cpu = make_memory(
      {pool1_dst_tz, memory::data_type::f32, memory::format_tag::nchw},
      cpu_engine);
  auto pool1_m_dst_gpu = make_memory(
      {pool1_dst_tz, memory::data_type::f32, memory::format_tag::nchw},
      gpu_engine);
  end = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
  std::cout << "Memory creation time: " << (end - begin) << " ms." << std::endl;
  begin = end;
  // ****************************************************************************************
  // // Reorder CPU -> GPU
  auto r1 = reorder(pool1_m_src_cpu, pool1_m_src_gpu);
  // Relu
  auto relu_desc =
      eltwise_forward::desc(prop_kind::forward, algorithm::eltwise_relu,
                            pool1_m_src_gpu.get_desc(), 0.0f);
  auto relu_pd = eltwise_forward::primitive_desc(relu_desc, gpu_engine);
  auto relu = eltwise_forward(relu_pd);
  // Pool
  auto pool1_desc = pooling_forward::desc(
      prop_kind::forward_inference, algorithm::pooling_max,
      pool1_m_src_gpu.get_desc(), pool1_m_dst_gpu.get_desc(), pool1_strides,
      pool1_kernel, pool1_padding, pool1_padding);
  auto pool1_pd = pooling_forward::primitive_desc(pool1_desc, gpu_engine);
  auto pool1 = pooling_forward(pool1_pd);
  /// reorder GPU -> CPU
  auto r2 = reorder(pool1_m_dst_gpu, pool1_m_dst_cpu);
  // ****************************************************************************************
  // //
  end = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
  std::cout << "Create Primitives time: " << (end - begin) << " ms."
            << std::endl;
  begin = end;

  // ****************************************************************************************
  // // wrap source data from CPU to GPU
  r1.execute(stream_gpu, pool1_m_src_cpu, pool1_m_src_gpu);
  // Execute ReLU + MaxPool on a GPU stream
  relu.execute(stream_gpu, {{DNNL_ARG_SRC, pool1_m_src_gpu},
                            {DNNL_ARG_DST, pool1_m_src_gpu}});
  pool1.execute(stream_gpu, {{DNNL_ARG_SRC, pool1_m_src_gpu},
                             {DNNL_ARG_DST, pool1_m_dst_gpu}});
  // Get result data from GPU to CPU
  r2.execute(stream_gpu, pool1_m_dst_gpu, pool1_m_dst_cpu);
  stream_gpu.wait();
  // ****************************************************************************************
  // //
  end = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
  std::cout << "Execution time: " << (end - begin) << " ms." << std::endl;
}

int main() {
  cross_engine_reorder();
  return 1;
}

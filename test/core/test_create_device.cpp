#undef NDEBUG
#include "../../src/core/device.hpp"
#include "../test_helper.hpp"
#include "dnnl.hpp"
#include <cassert>

using namespace dnnl;
using namespace std;

int test_create_device_cpu() {
  const auto name = rand_alnum_str(10);
  const auto engine_kind = "cpu";
  const auto device_id = 0;
  const auto num_streams = rand_int32_number(1, 10);
  Device dev(name, engine_kind, device_id, num_streams);
  assert(dev.name == name);
  assert(dev.get_engine().get_kind() == engine::kind::cpu);
  return EXIT_SUCCESS;
}

int test_create_device_cpu_throws_on_zero_streams() {
  const auto name = rand_alnum_str(10);
  const auto engine_kind = "cpu";
  const auto num_streams = -1;
  const auto expected =
      "Cannot create Device instance due to negative number of streams.";
  try {
    Device dev(name, engine_kind, 0, num_streams);
  } catch (runtime_error &error) {
    const string &msg = error.what();
    if (msg.compare(expected) != 0)
      return EXIT_FAILURE;
    return EXIT_SUCCESS;
  }
  return EXIT_FAILURE;
}

int main() {
  assert(test_create_device_cpu() == EXIT_SUCCESS);
  assert(test_create_device_cpu_throws_on_zero_streams() == EXIT_SUCCESS);
  return 1;
}
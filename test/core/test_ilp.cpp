#include <cassert>
#include <iostream>
#include <map>
#include <math.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#undef NDEBUG
#ifdef HAS_CBC
#include "../../src/core/ilp_solver_cbc.cpp"
#endif
#ifdef HAS_GUROBI
#include "../../src/core/ilp_solver_grb.cpp"
#endif
#if !defined(HAS_CBC) && !defined(HAS_GUROBI)
#include "../../src/core/ilp_solver.hpp"
#endif

using namespace std;

vector<double> solve_ilp(string model_name, vector<pair<string, edge>> &edges,
                         vector<string> &devices,
                         vector<vector<float>> &compute_costs,
                         vector<float> &memory_costs, matrix copy_costs,
                         vector<float> budget, vector<float> ram,
                         const string mode) {
  vector<double> results;
  const int verbose = 0;
  if (mode == "GUROBI") {
#ifdef HAS_GUROBI
    auto ilp = ILP_Solver_GRB(model_name, "", "", edges, devices, compute_costs,
                              memory_costs, copy_costs, budget, ram, verbose);
    ilp.solve();
    if (verbose > 0) {
      ilp.printResults();
    }
    results.push_back(ilp.get_minimal_compute_costs());
    results.push_back(ilp.get_minimal_memory());
    results.push_back(ilp.get_maximal_memory());
#else
    throw runtime_error("not compiled with GUROBI");
#endif
  } else if (mode == "CBC") {
#ifdef HAS_CBC
    auto ilp = ILP_Solver_CBC(model_name, "", "", edges, devices, compute_costs,
                              memory_costs, copy_costs, budget, ram, verbose);
    ilp.solve();
    if (verbose > 0) {
      ilp.printResults();
    }
    results.push_back(ilp.get_minimal_compute_costs());
    results.push_back(ilp.get_minimal_memory());
    results.push_back(ilp.get_maximal_memory());
#else
    throw runtime_error("not compiled with CBC");
#endif
  } else {
    throw runtime_error("unsupported Solver mode!");
  }
  return results;
}

int ilp_simple_costs_very_tiny_linear(const string mode, const float eps) {
  vector<float> budget = vector<float>{5e6};
  vector<float> ram = vector<float>{1.0};
  auto compute_costs = vector<vector<float>>();
  compute_costs.push_back(vector<float>{1, 10});
  auto memory_costs = vector<float>{1e6, 4e6};
  assert(compute_costs[0].size() == memory_costs.size());
  vector<pair<string, edge>> edges = {{"0->1", edge(0, 1)}};
  vector<string> devices = {"cpu_0"};
  matrix copy_costs = matrix(edges.size(), devices.size(), devices.size(), 0);
  vector<double> results =
      solve_ilp("ilp_simple_costs_very_tiny_linear_" + mode, edges, devices,
                compute_costs, memory_costs, copy_costs, budget, ram, mode);
  assert(results.size() == 3);
  assert(fabs(results[0] - 11) < eps);
  assert(fabs(results[1] - 1e6) < eps);
  assert(fabs(results[2] - 5e6) < eps);
  return 0;
}

int ilp_simple_costs_tiny_linear(const string mode, const float eps) {
  vector<float> budget = vector<float>{5e6};
  vector<float> ram = vector<float>{1.0};
  auto compute_costs = vector<vector<float>>();
  compute_costs.push_back(vector<float>{1, 10, 5});
  auto memory_costs = vector<float>{1e6, 4e6, 1e6};
  assert(compute_costs[0].size() == memory_costs.size());
  vector<pair<string, edge>> edges = {{"0->1", edge(0, 1)},
                                      {"1->2", edge(1, 2)}};
  vector<string> devices = {"cpu_0"};
  matrix copy_costs = matrix(edges.size(), devices.size(), devices.size(), 0);
  vector<double> results =
      solve_ilp("ilp_simple_costs_tiny_linear_" + mode, edges, devices,
                compute_costs, memory_costs, copy_costs, budget, ram, mode);
  assert(results.size() == 3);
  assert(fabs(results[0] - 16) < eps);
  assert(fabs(results[1] - 1e6) < eps);
  assert(fabs(results[2] - 5e6) < eps);
  return 0;
}

int ilp_simple_costs_linear(const string mode, const float eps) {
  vector<float> budget = vector<float>{5e6};
  vector<float> ram = vector<float>{1.0};
  auto compute_costs = vector<vector<float>>();
  compute_costs.push_back(vector<float>{1, 10, 5, 1, 4, 2, 1});
  auto memory_costs = vector<float>{1e6, 4e6, 1e6, 1e6, 3e6, 1.5e6, 1e6};
  assert(compute_costs[0].size() == memory_costs.size());
  vector<pair<string, edge>> edges = {
      {"0->1", edge(0, 1)}, {"1->2", edge(1, 2)}, {"2->3", edge(2, 3)},
      {"3->4", edge(3, 4)}, {"4->5", edge(4, 5)}, {"5->6", edge(5, 6)}};
  vector<string> devices = {"cpu_0"};
  matrix copy_costs = matrix(edges.size(), devices.size(), devices.size(), 0);
  vector<double> results =
      solve_ilp("ilp_simple_costs_linear_" + mode, edges, devices,
                compute_costs, memory_costs, copy_costs, budget, ram, mode);
  assert(results.size() == 3);
  assert(fabs(results[0] - 24) < eps);
  assert(fabs(results[1] - 1e6) < eps);
  assert(fabs(results[2] - 5e6) < eps);
  return 0;
}

int ilp_simple_costs_skip(const string mode, const float eps) {
  // needs more budget due to skip connection ...
  vector<float> budget = vector<float>{6e6};
  vector<float> ram = vector<float>{1.0};
  auto compute_costs = vector<vector<float>>();
  compute_costs.push_back(vector<float>{1, 10, 5, 1, 4, 2, 1});
  auto memory_costs = vector<float>{1e6, 4e6, 1e6, 1e6, 3e6, 1.5e6, 1e6};
  assert(compute_costs[0].size() == memory_costs.size());
  vector<pair<string, edge>> edges = {
      {"0->1", edge(0, 1)}, {"1->2", edge(1, 2)}, {"2->3", edge(2, 3)},
      {"3->4", edge(3, 4)}, {"4->5", edge(4, 5)}, {"5->6", edge(5, 6)},
      {"1->3", edge(1, 3)}, {"2->6", edge(2, 6)}};
  vector<string> devices = {"cpu_0"};
  matrix copy_costs = matrix(edges.size(), devices.size(), devices.size(), 0);
  vector<double> results =
      solve_ilp("ilp_simple_costs_skip_" + mode, edges, devices, compute_costs,
                memory_costs, copy_costs, budget, ram, mode);
  assert(results.size() == 3);
  assert(fabs(results[0] - 24) < eps);
  assert(fabs(results[1] - 1e6) < eps);
  // maximum memory when computing node 3:
  // edge(1, 2) + edge(2, 3) + edge(1, 3)
  // -> keep inputs m[1] and m[2], output m[3] in memory
  // 4 + 1 + 1 = 6
  assert(fabs(results[2] - 6e6) < eps);
  return 0;
}

int ilp_multi_costs_very_tiny_linear(const string mode, const float eps) {
  vector<float> budget = vector<float>{5e6, 5e6};
  vector<float> ram = vector<float>{1.0, 1.0};
  auto compute_costs = vector<vector<float>>();
  compute_costs.push_back(vector<float>{1, 10});
  compute_costs.push_back(vector<float>{10, 1});
  auto memory_costs = vector<float>{1e6, 4e6};
  assert(compute_costs[0].size() == memory_costs.size());
  vector<pair<string, edge>> edges = {{"0->1", edge(0, 1)}};
  vector<string> devices = {"cpu_0", "gpu_0"};
  matrix copy_costs = matrix(edges.size(), devices.size(), devices.size(), 0);
  vector<double> results =
      solve_ilp("ilp_multi_costs_very_tiny_linear_" + mode, edges, devices,
                compute_costs, memory_costs, copy_costs, budget, ram, mode);
  assert(results.size() == 3);
  assert(fabs(results[0] - 2) < eps);
  assert(fabs(results[1] - 0e6) < eps);
  // maximum memory between 4e6 and 5e6,
  // depending on weather free or keep 1st node
  // (since the budget is sufficient)
  assert(fabs(results[2] - 4e6) < eps + 1e6);
  return 0;
}

int ilp_multi_costs_very_tiny_linear_copy_costs(const string mode,
                                                const float eps) {
  vector<float> budget = vector<float>{5e6, 5e6};
  vector<float> ram = vector<float>{1.0, 1.0};
  auto compute_costs = vector<vector<float>>();
  compute_costs.push_back(vector<float>{1, 10});
  compute_costs.push_back(vector<float>{10, 1});
  auto memory_costs = vector<float>{1e6, 4e6};
  assert(compute_costs[0].size() == memory_costs.size());
  vector<pair<string, edge>> edges = {{"0->1", edge(0, 1)}};
  vector<string> devices = {"cpu_0", "gpu_0"};
  /*
  copy costs
  |- - |
  |0 3 |
  |8 0 |
  |- - |
  */
  matrix copy_costs = matrix(edges.size(), devices.size(), devices.size(), 0);
  copy_costs.set(0, 0, 1, 3);
  copy_costs.set(0, 1, 0, 8);
  copy_costs.print();

  // best option: compute n0 on d0, copy t0 from d0 to d1, compute n1 on d1
  vector<double> results = solve_ilp(
      "ilp_multi_costs_very_tiny_linear_copy_costs_" + mode, edges, devices,
      compute_costs, memory_costs, copy_costs, budget, ram, mode);
  assert(results.size() == 3);
  assert(fabs(results[0] - 5) < eps);
  // 2 + copy_tensor0_from_0_to_1 = 2 + 3 = 5
  assert(fabs(results[1] - 0e6) < eps);
  // maximum memory between 4e6 and 5e6,
  // depending on weather free or keep 1st node
  // (since the budget is sufficient)
  assert(fabs(results[2] - 4e6) < eps + 1e6);
  return 0;
}

int ilp_multi_costs_tiny_linear(const string mode, const float eps) {
  vector<float> budget = vector<float>{5e6, 5e6};
  vector<float> ram = vector<float>{1.0, 1.0};
  auto compute_costs = vector<vector<float>>();
  compute_costs.push_back(vector<float>{1, 10, 1});
  compute_costs.push_back(vector<float>{10, 1, 10});
  auto memory_costs = vector<float>{1e6, 4e6, 1e6};
  assert(compute_costs[0].size() == memory_costs.size());
  vector<pair<string, edge>> edges = {{"0->1", edge(0, 1)},
                                      {"1->2", edge(1, 2)}};
  vector<string> devices = {"cpu_0", "gpu_0"};
  // if memory costs == 0 -> optimum == 3
  /*
  copy costs
  |- - |
  |0 3 |
  |8 0 |
  |- - |
  |- - |
  |0 5 |
  |1 0 |
  |- - |
  */
  matrix copy_costs = matrix(edges.size(), devices.size(), devices.size(), 0);
  copy_costs.set(0, 0, 1, 3);
  copy_costs.set(0, 1, 0, 8);
  copy_costs.set(1, 0, 1, 5);
  copy_costs.set(1, 1, 0, 1);
  vector<double> results =
      solve_ilp("ilp_multi_costs_tiny_linear_" + mode, edges, devices,
                compute_costs, memory_costs, copy_costs, budget, ram, mode);
  assert(results.size() == 3);
  // 3 + copy_costs + 3 + 1
  assert(fabs(results[0] - 7) < eps);
  assert(fabs(results[1] - 0e6) < eps);
  assert(fabs(results[2] - 5e6) < eps);
  return 0;
}

int ilp_multi_costs_linear(const string mode, const float eps) {
  /*
  compute everything on d0: costs 16
  compute 4th node on d1 has costs 1 -> can we switch?
  switching has copy costs:
    copy_costs.at(2, 0, 1) which is 2 and
    copy_costs.at(3, 1, 0) which is 3
    -> switching costs 4 which is less than staying (10)!
    total costs = compute costs +    copy costs
                =    (7 * 1)    +      (2 + 3)     = 12
  */
  vector<float> budget = vector<float>{5e6, 5e6};
  vector<float> ram = vector<float>{1.0, 1.0};
  auto compute_costs = vector<vector<float>>();
  compute_costs.push_back(vector<float>{1, 1, 1, 10, 1, 1, 1});
  compute_costs.push_back(vector<float>{10, 10, 10, 1, 10, 10, 10});
  auto memory_costs = vector<float>{1e6, 4e6, 1e6, 1e6, 3e6, 1.5e6, 1e6};
  assert(compute_costs[0].size() == memory_costs.size());
  vector<pair<string, edge>> edges = {
      {"0->1", edge(0, 1)}, {"1->2", edge(1, 2)}, {"2->3", edge(2, 3)},
      {"3->4", edge(3, 4)}, {"4->5", edge(4, 5)}, {"5->6", edge(5, 6)}};
  vector<string> devices = {"cpu_0", "gpu_0"};
  /*
  copy costs
  |- - |
  |0 6 |
  |5 0 |
  |- - |
  |- - |
  |0 3 |
  |8 0 |
  |- - |
  |- - |
  |0 2 | -
  |9 0 |
  |- - |
  |- - |
  |0 8 |
  |3 0 | -
  |- - |
  |- - |
  |0 7 |
  |8 0 |
  |- - |
  |- - |
  |0 5 |
  |1 0 |
  |- - |
  */
  matrix copy_costs = matrix(edges.size(), devices.size(), devices.size(), 0);
  copy_costs.set(0, 0, 1, 6);
  copy_costs.set(0, 1, 0, 5);
  copy_costs.set(1, 0, 1, 3);
  copy_costs.set(1, 1, 0, 8);
  copy_costs.set(2, 0, 1, 2);
  copy_costs.set(2, 1, 0, 9);
  copy_costs.set(3, 0, 1, 8);
  copy_costs.set(3, 1, 0, 3);
  copy_costs.set(4, 0, 1, 7);
  copy_costs.set(4, 1, 0, 8);
  copy_costs.set(5, 0, 1, 5);
  copy_costs.set(5, 1, 0, 1);
  vector<double> results =
      solve_ilp("ilp_multi_costs_linear_" + mode, edges, devices, compute_costs,
                memory_costs, copy_costs, budget, ram, mode);
  assert(results.size() == 3);
  assert(fabs(results[0] - 12) < eps);
  assert(fabs(results[1] - 0e6) < eps);
  assert(fabs(results[2] - 5e6) < eps);
  return 0;
}

int ilp_multi_costs_skip(const string mode, const float eps) {
  vector<float> budget = vector<float>{6e6, 6e6};
  vector<float> ram = vector<float>{1.0, 1.0};
  auto compute_costs = vector<vector<float>>();
  compute_costs.push_back(vector<float>{1, 10, 5, 1, 4, 2, 1});
  compute_costs.push_back(vector<float>{0.5, 2, 1.5, 0.5, 7, 4, 2});
  // sequence d1,  d1,  d1,   d1,  d0, d0, d0 is optimal
  //          0.5 + 2 + 1.5 + 0.5 + 4 + 2 + 1 = 11.5
  auto memory_costs = vector<float>{1e6, 4e6, 1e6, 1e6, 3e6, 1.5e6, 1e6};
  assert(compute_costs[0].size() == memory_costs.size());
  assert(compute_costs[1].size() == memory_costs.size());
  vector<pair<string, edge>> edges = {
      {"0->1", edge(0, 1)}, {"1->2", edge(1, 2)}, {"2->3", edge(2, 3)},
      {"3->4", edge(3, 4)}, {"4->5", edge(4, 5)}, {"5->6", edge(5, 6)},
      {"1->3", edge(1, 3)}, {"2->6", edge(2, 6)}};
  vector<string> devices = {"cpu_0", "gpu_0"};
  matrix copy_costs = matrix(edges.size(), devices.size(), devices.size(), 0);
  vector<double> results =
      solve_ilp("ilp_multi_costs_skip_" + mode, edges, devices, compute_costs,
                memory_costs, copy_costs, budget, ram, mode);
  assert(results.size() == 3);
  assert(fabs(results[0] - 11.5) < eps);
  assert(fabs(results[1] - 0e6) < eps);
  return 0;
}

int ilp_multi_costs_skip_copy_costs(const string mode, const float eps) {
  /*
  compute everything on d0: costs 16
  compute 4th node on d1 has costs 1 -> can we switch?
  switching has copy costs:
    copy_costs.at(2, 0, 1) which is 2 and
    copy_costs.at(1, 0, 1) which is 3 and
    copy_costs.at(3, 1, 0) which is 3
    -> switching costs 9 which is less than staying (10)!
    total costs = compute costs +    copy costs
                =    (7 * 1)    +      (2 + 3 + 3)   = 15
  */
  vector<float> budget = vector<float>{5e6, 5e6};
  vector<float> ram = vector<float>{1.0, 1.0};
  auto compute_costs = vector<vector<float>>();
  compute_costs.push_back(vector<float>{1, 1, 1, 10, 1, 1, 1});
  compute_costs.push_back(vector<float>{10, 10, 10, 1, 10, 10, 10});
  auto memory_costs = vector<float>{1e6, 4e6, 1e6, 1e6, 3e6, 1.5e6, 1e6};
  assert(compute_costs[0].size() == memory_costs.size());
  assert(compute_costs[1].size() == memory_costs.size());
  vector<pair<string, edge>> edges = {
      {"0->1", edge(0, 1)}, {"1->2", edge(1, 2)}, {"2->3", edge(2, 3)},
      {"3->4", edge(3, 4)}, {"4->5", edge(4, 5)}, {"5->6", edge(5, 6)},
      {"1->3", edge(1, 3)}, {"2->6", edge(2, 6)}};
  vector<string> devices = {"cpu_0", "gpu_0"};
  // best sequence: d0 d0 d0 d1 d0 d0 d0
  // costs: 7 + 2+3+4
  /*
  copy costs
  |- - |
  |0 6 |
  |5 0 |
  |- - |
  |- - |
  |0 3 |
  |8 0 |
  |- - |
  |- - |
  |0 2 |
  |9 0 |
  |- - |
  |- - |
  |0 8 |
  |3 0 |
  |- - |
  |- - |
  |0 7 |
  |8 0 |
  |- - |
  |- - |
  |0 5 |
  |1 0 |
  |- - |
  |- - |
  |0 7 |
  |4 0 |
  |- - |
  |- - |
  |0 2 |
  |1 0 |
  |- - |
  */
  matrix copy_costs = matrix(edges.size(), devices.size(), devices.size(), 0);
  copy_costs.set(0, 0, 1, 6);
  copy_costs.set(0, 1, 0, 5);
  copy_costs.set(1, 0, 1, 3);
  copy_costs.set(1, 1, 0, 8);
  copy_costs.set(2, 0, 1, 2);
  copy_costs.set(2, 1, 0, 9);
  copy_costs.set(3, 0, 1, 8);
  copy_costs.set(3, 1, 0, 3);
  copy_costs.set(4, 0, 1, 7);
  copy_costs.set(4, 1, 0, 8);
  copy_costs.set(5, 0, 1, 5);
  copy_costs.set(5, 1, 0, 1);
  copy_costs.set(6, 0, 1, 7);
  copy_costs.set(6, 1, 0, 4);
  copy_costs.set(7, 0, 1, 2);
  copy_costs.set(7, 1, 0, 1);
  vector<double> results =
      solve_ilp("ilp_multi_costs_skip_copy_costs_" + mode, edges, devices,
                compute_costs, memory_costs, copy_costs, budget, ram, mode);
  assert(results.size() == 3);
  assert(fabs(results[0] - 16) < eps);
  assert(fabs(results[1] - 0e6) < eps);
  assert(fabs(results[2] - 5e6) < eps);
  return 0;
}

int performAllTests(const string mode) {
  const float eps = 1e-6;
  ilp_simple_costs_linear(mode, eps);
  ilp_simple_costs_skip(mode, eps);
  ilp_multi_costs_very_tiny_linear_copy_costs(mode, eps);
  ilp_multi_costs_linear(mode, eps);
  ilp_multi_costs_skip(mode, eps);
  ilp_multi_costs_skip_copy_costs(mode, eps);
  return 0;
}

int main() {
#ifdef HAS_GUROBI
  performAllTests("GUROBI");
  cout << "GUROBI: All Tests passed." << endl;
#endif
#ifdef HAS_CBC
  performAllTests("CBC");
  cout << "CBC All Tests passed." << endl;
#endif
  return 0;
}
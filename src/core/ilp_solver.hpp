#ifndef XENGINE_ILP_SOLVER_HPP
#define XENGINE_ILP_SOLVER_HPP

#include "edge.hpp"
#include "matrix.hpp"
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

vector<edge> get_edges_linear(const size_t T) {
  auto edges = std::vector<edge>();
  edges.reserve(T);
  for (size_t t = 0; t < T - 1; ++t) {
    edges.push_back(edge(t, t + 1));
  }
  edges.push_back(edge(0, T - 1));
  return edges;
}

string idx2str(const size_t i, const size_t width = 6) {
  stringstream ss;
  ss << setw(width) << setfill('0') << i;
  return ss.str();
}

string idx_to_string(const size_t d, const size_t i, const size_t j) {
  return "[" + idx2str(d, 3) + "," + idx2str(i) + "," + idx2str(j) + "]";
}

class ILP_Solver {
public:
  ILP_Solver(string model_name, string mpsfile, string logfile,
             vector<pair<string, edge>> edges, vector<string> devices,
             vector<vector<float>> compute_costs, vector<float> memory_costs,
             matrix copy_costs, vector<float> budget, vector<float> ram,
             const int verbose = 0);
  ~ILP_Solver();
  virtual int solve() = 0;
  void printResults();
  matrix get_R();
  matrix get_S();
  double get_minimal_compute_costs();
  double get_minimal_memory();
  double get_maximal_memory();

protected:
  vector<size_t> _successors(const size_t idx);
  vector<pair<size_t, size_t>> _predecessors(const size_t idx);
  size_t _max_num_hazards(const size_t t, const size_t d, const size_t i,
                          const size_t k);
  string _model_name;
  vector<float> _budget;
  vector<float> _ram;
  vector<vector<float>> _compute_costs;
  matrix _copy_costs;
  vector<float> _memory_costs;
  vector<pair<string, edge>> _edges;
  vector<string> _devices;
  matrix *_R_matrix;
  matrix *_S_matrix;
  matrix *_Z_matrix;
  matrix *_U_matrix;
  matrix *_F_matrix;
  double _objective_value;
  double _minimal_memory;
  double _maximal_memory;
  int _verbose;
  string _mpsfile;
  string _logfile;
};
#endif
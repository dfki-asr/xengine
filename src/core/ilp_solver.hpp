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

class Marker {
public:
  Marker() {
    name = "";
    mode = "";
    values = vector<pair<string, float>>();
    indices = vector<size_t>();
  }
  Marker(const string m, const string n, const vector<size_t> i) {
    name = n;
    mode = m;
    values = vector<pair<string, float>>();
    indices = i;
  }
  string name;
  string mode;
  vector<pair<string, float>> values;
  vector<size_t> indices;
};

class Constraint {
public:
  Constraint() {
    name = "";
    mode = "";
    rhs = 0.0;
  }
  Constraint(const string m, const string n, const float v) {
    name = n;
    mode = m;
    rhs = v;
  }
  string name;
  string mode;
  float rhs;
};

class QuadObj {
public:
  QuadObj() {
    var1 = "";
    var2 = "";
    coeff = 0;
  }
  QuadObj(const string v1, const string v2, const float c) {
    var1 = v1;
    var2 = v2;
    coeff = c;
  }
  string var1;
  string var2;
  float coeff;
};

class ILP_Solver {
public:
  ILP_Solver(string model_name, string mpsfile, string logfile,
             vector<pair<string, edge>> edges, vector<string> devices,
             vector<vector<float>> compute_costs, vector<float> memory_costs,
             matrix copy_costs, vector<float> budget, vector<float> ram,
             const int verbose = 0);
  ~ILP_Solver();
  int defineProblemAsMPS();
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

private:
  string _rows;
  string _columns;
  string _rhs;
  map<string, Marker> _marker;
  unordered_map<string, Constraint> _constraints;
  vector<string> _constraints_insertOrder;
  vector<QuadObj> _quad_obj;

  void _num_hazards(const size_t t, const size_t d, const size_t i,
                    const size_t k, const string c_ub, const string c_lb);

  void add_constraint(const string mode, const float rhs);
  void add_marker(const string mode, const string name, const string type,
                  const float value, const vector<size_t> indices);
  void add_row(const string name);
  void add_column(const string name);
};
#endif
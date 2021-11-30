#ifndef XENGINE_ILP_SOLVER_CBC_HPP
#define XENGINE_ILP_SOLVER_CBC_HPP

#include "CbcSolver.hpp"
#include "OsiClpSolverInterface.hpp"
#include "ilp_solver.hpp"

using namespace std;

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

class ILP_Solver_CBC : public ILP_Solver {
public:
  ILP_Solver_CBC(string model_name, string mpsfile, string logfile,
                 vector<pair<string, edge>> edges, vector<string> devices,
                 vector<vector<float>> compute_costs,
                 vector<float> memory_costs, matrix &copy_costs,
                 vector<float> budget, vector<float> ram,
                 const int verbose = 0);
  ~ILP_Solver_CBC();
  int solve();

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

  int defineProblem();
};
#endif
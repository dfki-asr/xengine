#ifndef XENGINE_ILP_SOLVER_GRB_HPP
#define XENGINE_ILP_SOLVER_GRB_HPP

#include "ilp_solver.hpp"

#include "gurobi_c++.h"

using namespace std;

class ILP_Solver_GRB : public ILP_Solver {
public:
  ILP_Solver_GRB(string model_name, string mpsfile, string logfile,
                 vector<pair<string, edge>> edges, vector<string> devices,
                 vector<vector<float>> compute_costs,
                 vector<float> memory_costs, matrix &copy_costs,
                 vector<float> budget, vector<float> ram,
                 const int verbose = 0);
  ~ILP_Solver_GRB();
  int solve();

private:
  GRBLinExpr _num_hazards(const size_t t, const size_t d, const size_t i,
                          const size_t k, GRBVar *R, GRBVar *S);

  int defineProblem();
};
#endif
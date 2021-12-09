#ifndef XENGINE_ILP_SOLVER_CBC_HPP
#define XENGINE_ILP_SOLVER_CBC_HPP

#include "CbcSolver.hpp"
#include "OsiClpSolverInterface.hpp"
#include "ilp_solver.hpp"

using namespace std;

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
};
#endif
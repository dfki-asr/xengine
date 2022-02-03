#ifndef XENGINE_ILP_SOLVER_CBC_CPP
#define XENGINE_ILP_SOLVER_CBC_CPP

#include "ilp_solver_cbc.hpp"
#include "ilp_solver.cpp"
#include <sstream>

using namespace std::placeholders;

ILP_Solver_CBC::ILP_Solver_CBC(string model_name, string mpsfile,
                               string logfile, vector<pair<string, edge>> edges,
                               vector<string> devices,
                               vector<vector<float>> compute_costs,
                               vector<float> memory_costs, matrix &copy_costs,
                               vector<float> budget, vector<float> ram,
                               const int verbose)
    : ILP_Solver(model_name, mpsfile, logfile, edges, devices, compute_costs,
                 memory_costs, copy_costs, budget, ram, verbose) {
  _mpsfile = mpsfile.empty() ? _model_name + "_cbc.mps" : mpsfile;
  _logfile = logfile.empty() ? _model_name + "_cbc.log" : logfile;
}
ILP_Solver_CBC::~ILP_Solver_CBC() {}

int ILP_Solver_CBC::solve() {
  try {
    OsiClpSolverInterface solver1;
    CbcModel model(solver1);
    CbcSolverUsefulData cbcData;
    cbcData.useSignalHandler_ = true;
    cbcData.noPrinting_ = false;
    CbcMain0(model, cbcData);
    if (_verbose > 0) {
      cout << "Read ilp model file " << _mpsfile << endl;
    }
    int argc = 2;
    const char *argv[argc] = {"cbc", _mpsfile.c_str()};
    model.setDblParam(CbcModel::CbcMaximumSeconds, 1200.0);
    CbcMain1(argc, argv, model, cbcData);

    // Print solution
    if (!model.status()) {
      _objective_value = model.getObjValue();
      int numberColumns = model.solver()->getNumCols();
      const double *solution = model.solver()->getColSolution();
      const size_t T = _compute_costs[0].size();
      const size_t E = _edges.size();
      const size_t D = _devices.size();
      double min_mem = 1e12;
      double max_mem = 0;
      size_t offset_F = 0;
      size_t offset_R = offset_F + D * T * E;
      size_t offset_S = offset_R + D * T * T;
      size_t offset_U = offset_S + D * T * T;
      size_t offset_Z = offset_U + D * T * T;
      size_t d, t, i, e;
      for (d = 0; d < D; ++d) {
        for (t = 0; t < T; ++t) {
          for (i = 0; i < T; ++i) {
            const size_t idx = (d * T * T) + (t * T) + i;
            _R_matrix->set(idx, solution[offset_R + idx]);
            _S_matrix->set(idx, solution[offset_S + idx]);
            _Z_matrix->set(idx, solution[offset_Z + idx]);
            double u = solution[offset_U + idx];
            _U_matrix->set(idx, u);
            if (u < min_mem) {
              min_mem = u;
            }
            if (u > max_mem) {
              max_mem = u;
            }
          }
          for (e = 0; e < E; ++e) {
            const size_t idx = (d * T * E) + (t * E) + e;
            _F_matrix->set(idx, solution[offset_F + idx]);
          }
        }
      }
      _minimal_memory = min_mem;
      _maximal_memory = max_mem;
    }
  } catch (...) {
    cout << "Exception during optimization" << endl;
    throw runtime_error("No solution for " + _model_name);
  }
  return 0;
}
#endif
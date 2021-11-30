#ifndef XENGINE_ILP_SOLVER_CPP
#define XENGINE_ILP_SOLVER_CPP

#include "ilp_solver.hpp"

ILP_Solver::ILP_Solver(string model_name, string mpsfile, string logfile,
                       vector<pair<string, edge>> edges, vector<string> devices,
                       vector<vector<float>> compute_costs,
                       vector<float> memory_costs, matrix copy_costs,
                       vector<float> budget, vector<float> ram,
                       const int verbose) {
  _model_name = model_name;
  _mpsfile = mpsfile.empty() ? _model_name + ".mps" : mpsfile;
  _logfile = logfile.empty() ? _model_name + ".log" : logfile;
  _edges = edges;
  _devices = devices;
  _compute_costs = compute_costs;
  _memory_costs = memory_costs;
  _copy_costs = copy_costs;
  _budget = budget;
  _ram = ram;
  _objective_value = -1;
  _minimal_memory = -1;
  _maximal_memory = -1;
  _R_matrix = nullptr;
  _S_matrix = nullptr;
  _Z_matrix = nullptr;
  _U_matrix = nullptr;
  _F_matrix = nullptr;
  _verbose = verbose;
  // init
  const size_t T = _compute_costs[0].size();
  const size_t E = _edges.size();
  const size_t D = _devices.size();
  size_t binary = 1;
  _R_matrix = new matrix(D, T, T, binary);
  _S_matrix = new matrix(D, T, T, binary);
  _Z_matrix = new matrix(D, T, T, binary);
  _F_matrix = new matrix(D, T, E, binary);
  binary = 0;
  _U_matrix = new matrix(D, T, T, binary);
}

ILP_Solver::~ILP_Solver() {
  if (_R_matrix != nullptr) {
    delete _R_matrix;
  }
  if (_S_matrix != nullptr) {
    delete _S_matrix;
  }
  if (_Z_matrix != nullptr) {
    delete _Z_matrix;
  }
  if (_U_matrix != nullptr) {
    delete _U_matrix;
  }
  if (_F_matrix != nullptr) {
    delete _F_matrix;
  }
}

matrix ILP_Solver::get_R() { return *_R_matrix; }
matrix ILP_Solver::get_S() { return *_S_matrix; }
double ILP_Solver::get_minimal_compute_costs() { return _objective_value; }
double ILP_Solver::get_minimal_memory() { return _minimal_memory; }
double ILP_Solver::get_maximal_memory() { return _maximal_memory; }

vector<size_t> ILP_Solver::_successors(const size_t idx) {
  vector<size_t> succ;
  for (auto e : _edges) {
    if (e.second.get_u() == idx) {
      succ.push_back(e.second.get_v());
    }
  }
  return succ;
}

vector<pair<size_t, size_t>> ILP_Solver::_predecessors(const size_t idx) {
  vector<pair<size_t, size_t>> pred;
  for (auto e_idx = 0; e_idx < _edges.size(); ++e_idx) {
    edge e = _edges[e_idx].second;
    if (e.get_v() == idx) {
      pred.push_back(make_pair(e.get_u(), e_idx));
    }
  }
  return pred;
}

size_t ILP_Solver::_max_num_hazards(const size_t t, const size_t d,
                                    const size_t i, const size_t k) {
  const size_t T = _compute_costs[d].size();
  size_t num_uses_after_k = 0;
  for (auto j : _successors(i)) {
    if (j > k) {
      num_uses_after_k += 1;
    }
  }
  if (t + 1 < T) {
    num_uses_after_k += 2;
  } else {
    num_uses_after_k += 1;
  }
  return num_uses_after_k;
}

void ILP_Solver::printResults() {
  if (_verbose > 0) {
    cout << "\nSolution with computational costs: " << _objective_value
         << ", memory consumption: min " << _minimal_memory << ", peak "
         << _maximal_memory << endl;
    cout << "\nR" << endl;
    _R_matrix->print();
    cout << "\nS" << endl;
    _S_matrix->print();
  }
  if (_verbose > 1) {
    cout << "\nZ" << endl;
    _Z_matrix->print();
    cout << "\nU" << endl;
    _U_matrix->print();
    cout << "\nF" << endl;
    _F_matrix->print();
  }
}

int getEdgeIndexFromName(vector<pair<string, edge>> &edges,
                         const string edgeName) {
  for (size_t edgeID = 0; edgeID < edges.size(); edgeID++) {
    if (edges[edgeID].first == edgeName) {
      return edgeID;
    }
  }
  cout << "Edge " + edgeName + " was not found!" << endl;
  throw runtime_error("Edge " + edgeName + " was not found!");
}
#endif
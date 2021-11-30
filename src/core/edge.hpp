#ifndef XENGINE_EDGE_HPP
#define XENGINE_EDGE_HPP

using namespace std;

class edge {
public:
  edge(size_t u, size_t v) {
    _u = u;
    _v = v;
  }
  std::string as_string() {
    return "(" + std::to_string(_u) + ", " + std::to_string(_v) + ")";
  }
  size_t get_u() { return _u; }
  size_t get_v() { return _v; }

private:
  size_t _u;
  size_t _v;
};
#endif
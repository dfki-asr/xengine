#ifndef XENGINE_MATRIX_HPP
#define XENGINE_MATRIX_HPP

using namespace std;

class matrix {
public:
  matrix() : _channels(0), _rows(0), _cols(0), _binary(0) {
    vals = vector<float>();
  }
  matrix(size_t channels, size_t rows, size_t cols, size_t binary = 1)
      : _channels(channels), _rows(rows), _cols(cols) {
    vals = vector<float>(channels * rows * cols, 0.0);
    _binary = binary;
  }
  float at(const size_t idx) { return vals[idx]; }
  float at(const size_t i, const size_t j) { return vals[i * _cols + j]; }
  float at(const size_t c, const size_t i, const int j) {
    return vals[(c * _rows * _cols) + (i * _cols) + j];
  }
  void set(const size_t idx, float value) { vals[idx] = value; }
  void set(const size_t i, const size_t j, float value) {
    vals[i * _cols + j] = value;
  }
  void set(const size_t c, const size_t i, const size_t j, float value) {
    vals[(c * _rows * _cols) + (i * _cols) + j] = value;
  }

  size_t get_channels() { return _channels; }
  size_t get_rows() { return _rows; }
  size_t get_cols() { return _cols; }
  size_t get_size() { return _channels * _rows * _cols; }

  void print() {
    const string border = "|";
    const string frame = "-";
    const string one = "\u2588";
    const string zero = " ";
    for (size_t c = 0; c < _channels; ++c) {
      cout << border;
      for (size_t i = 0; i < _cols; ++i) {
        cout << frame;
      }
      cout << border << endl;
      for (size_t t = 0; t < _rows; ++t) {
        cout << border;
        for (size_t i = 0; i < _cols; ++i) {
          if (_binary) {
            if (at(c, t, i) > 0) {
              cout << one;
            } else {
              cout << zero;
            }
          } else {
            cout << at(c, t, i) << " ";
          }
        }
        cout << border << endl;
      }
      cout << border;
      for (size_t i = 0; i < _cols; ++i) {
        cout << frame;
      }
      cout << border << endl;
    }
  }

private:
  size_t _channels, _rows, _cols;
  size_t _binary;
  vector<float> vals;
};
#endif
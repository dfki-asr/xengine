#ifndef XENGINE_TEST_HELPER_HPP
#define XENGINE_TEST_HELPER_HPP

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

int32_t rand_int32_number(int32_t start, int32_t end) {
  return rand() % end + start;
}

float rand_float_number(float start, float end) {
  return static_cast<float>(rand()) / end + start;
}

char rand_alnum() {
  char c;
  while (!std::isalnum(c = static_cast<char>(std::rand())))
    ;
  return c;
}

std::string rand_alnum_str(std::string::size_type sz) {
  std::string s;
  s.reserve(sz);
  generate_n(std::back_inserter(s), sz, rand_alnum);
  return s;
}

#endif
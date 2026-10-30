#include <usul.hpp>

float random_float() {
  std::random_device rd;
  std::mt19937 gen(rd());

  float min_val = 0.0F;
  float max_val = 10.0F;
  std::uniform_real_distribution<float> distr(min_val, max_val);

  float random_val = distr(gen);

  return random_val;
}
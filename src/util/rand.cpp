#include <random>
#include <util/rand.hpp>

float random_float() {
  std::random_device rd;
  std::mt19937 gen(rd());

  float min_val = 0.0f;
  float max_val = 10.0f;
  std::uniform_real_distribution<float> distr(min_val, max_val);

  float random_val = distr(gen);

  return random_val;
}
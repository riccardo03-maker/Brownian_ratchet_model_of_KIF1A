#include <cmath>
#include <iostream>
#include <random>
#include <vector>

int const temperature = 310;       /* Kelvin */
int D = 20000;                     // nanometer^2/second
int const k_on = 0, k_off = 0;     // 1/seconds
double const delta_t = 0.0000001;  // seconds
std::default_random_engine generator;
double mean, std_dev, accum = 0.;

double force(double x) {
  double z = std::fmod(x, 8);
  if (z >= 0 and z < 3) {
    return -1.3;
  }
  if (z >= 3 and z < 4) {
    return 3.5;
  }
  if (z >= 4 and z < 7) {
    return -0.7;
  }
  if (z >= 7 and z <= 8) {
    return 2.5;
  } else {
    return 0;
  }
}

double evolution(int const N, int D) {
  double t = 0, x = 0;
  int state = 0;
  std::normal_distribution<double> gaussian(0.0, delta_t);
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  for (int i = 0; i < N; ++i) {
    double brownian = gaussian(generator);
    if (state == 0) {
      x = x - (force(x) * delta_t * D) + (sqrt(2 * D) * brownian);
    }
    if (state == 1) {
      x = x + (sqrt(2 * D) * brownian);
    }
    t = t + delta_t;
    /*if (state == 0 and unif(generator) <= k_off * delta_t) {
      state = 1;
      D = 200;
      continue;
    }
    if (state == 1 and unif(generator) <= k_on * delta_t) {
      state = 0;
      D = 20000;
      continue;
    }*/
  }
  return x / t;
}

int main() {
  std::vector<double> velocities(10000, 0);
  for (int i = 0; i < 10000; ++i) {
    velocities[i] = evolution(1000000, D);
  }
  mean = std::accumulate(velocities.begin(), velocities.end(), 0.) /
         velocities.size();
  for (double x : velocities) {
    accum += (x - mean) * (x - mean);
  }
  std_dev = std::sqrt(accum / velocities.size());
  std::cout << "Mean:" << mean << "\n";
  std::cout << "Standard deviation" << std_dev << "\n";
}
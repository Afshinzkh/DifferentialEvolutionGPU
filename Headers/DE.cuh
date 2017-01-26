#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <curand.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <algorithm>
#include <vector>
#include <chrono>

#include "kernel.cuh"

// Main Differential Evolution Function
// void runDE(std::vector<double> const& mrktData);

class  DE{
  public:
    void runDE();
    DE(std::string m, const double dt);
    const double& getAlpha() const;
    const double& getBeta() const;
    const double& getSigma() const;
    const double& getError() const;
    const int& getIter() const;
    const double& getTime() const;
    void setMrktArray(std::vector<double> const& mrktData);
    const std::array<double, 9>& getMdlArray() const;

  private:
    std::string methodName;
    double alpha, beta, sigma, avgError;
    double dtTerm;
    int gens;
    double calTime;
    std::vector<double> crrntMonthMrktDataVec;
    std::array<double, 9> crrntMonthMdlDataArray;
};

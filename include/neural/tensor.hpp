#pragma once

#include <vector>
#include <xtensor/core/xmath.hpp>
#include <xtensor/containers/xarray.hpp>

using namespace std;

// there are 2 responsiblities of our comp graph, one is to allow dx
// and the other are to provide the dag for the backend.
// meaning we've got a frontend and a backend
// the frontend is what the programmers interact with
// the backend is how usul's ecosystem processes the DAG

class Tensor {

public:
  Tensor(xt::xarray<double> data,
         vector<Tensor> children = {},
         xt::xarray<double> gradients = {});
  xt::xarray<double> data;
  xt::xarray<double> gradients;

  vector<Tensor> children;
  function<void()> backwards;

  // addition
  Tensor operator+(Tensor other);
  // element wise
  Tensor operator*(Tensor other);
  // dot product
  Tensor operator^(Tensor other);
};

#pragma once

#include <memory>
#include <vector>
#include <xtensor/core/xmath.hpp>
#include <xtensor/containers/xarray.hpp>

using namespace std;

// there are 2 responsiblities of our comp graph, one is to allow dx
// and the other are to provide the dag for the backend.
// meaning we've got a frontend and a backend
// the frontend is what the programmers interact with
// the backend is how usul's ecosystem processes the DAG

class Tensor : public enable_shared_from_this<Tensor> {

public:
  Tensor(xt::xarray<double> data, vector<shared_ptr<Tensor>> children = {}, xt::xarray<double> gradients = {});
  xt::xarray<double> data;
  xt::xarray<double> gradients;

  

  vector<shared_ptr<Tensor>> children;
  function<void()> backwards;

  static shared_ptr<Tensor> create(xt::xarray<double> data, vector<shared_ptr<Tensor>> children = {}, xt::xarray<double> gradients = {});

  // addition
  shared_ptr<Tensor> operator+(Tensor &other);
  // element wise
  shared_ptr<Tensor> operator*(Tensor &other);
  // dot product
  shared_ptr<Tensor> operator^(Tensor &other);
  // simple relu
  shared_ptr<Tensor> ReLU();
  // simple sigmoid
  shared_ptr<Tensor> Sigmoid();

  shared_ptr<Tensor> getSharedPtr() {
    return shared_from_this();
  }
};

#pragma once
#include <cstddef>
#include <vector>
#include <xtensor/core/xmath.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/io/xio.hpp>
#include <memory>
#include <xtensor/utils/xutils.hpp>

struct Heart {
  // we need to move the data here.
  // these can be turned into a struct which can then be made into a shared pointer mutable std::shared_ptr<xt::xarray<double>> data;
  mutable xt::xarray<double> data;
  mutable xt::xarray<double> gradients = xt::zeros_like(data);
  std::function<void()> backwards;
  std::vector<std::shared_ptr<Heart>> children;
};

class Tensor {

public:
  // these many constructors, just so it doesnt have a sook when its not put into curly brace parameters.
  Tensor(xt::xarray<double> data);

  Tensor(xt::nested_initializer_list_t<double, 1> data)
      : Tensor(xt::xarray<double>(data)) {}

  Tensor(xt::nested_initializer_list_t<double, 2> data)
      : Tensor(xt::xarray<double>(data)) {}

  Tensor(xt::nested_initializer_list_t<double, 3> data)
      : Tensor(xt::xarray<double>(data)) {}

  Tensor(xt::nested_initializer_list_t<double, 4> data)
      : Tensor(xt::xarray<double>(data)) {}

  // our __str__ method overload.
  friend std::ostream &operator<<(std::ostream &os, const Tensor &t) {
    os << "Tensor(" << t.heart->data << ", " << "shape=" << t.shape() << ")\n";
    return os;
  }

  // define the type xtensor uses for its shape vectors
  using shapeType = typename xt::xarray<double>::shape_type;

  // we're using proxy pattern for seperating the view from the logic
  struct shapeView {
    const shapeType &raw_shape;
    friend std::ostream &operator<<(std::ostream &os, const shapeView &sv) {
      os << "(";
      for (size_t i = 0; i < sv.raw_shape.size(); i++) {
        os << sv.raw_shape[i];
        if (i < sv.raw_shape.size() - 1)
          os << ",";
      }
      if (sv.raw_shape.size() == 1) {
        os << ")";
      }
      return os;
    }

    // we are now going to allow implicit casting to vector or sizing lookups
    operator std::vector<size_t>() const {
      return {raw_shape.begin(), raw_shape.end()};
    }
    auto begin() const { return raw_shape.begin(); }
    auto end() const {
      return raw_shape.end();
    }
    size_t size() const { return raw_shape.size(); }
    size_t operator[](size_t index) const { return raw_shape[index]; }
  };

  // addition
  Tensor operator+(const Tensor &other) const;
  // addition scalar rhs
  Tensor operator+(double other) const;
  // addition scalar lhs
  friend Tensor operator+(double lhs, const Tensor &rhs);

  // element wise
  Tensor operator*(const Tensor &other) const;
  // element wise scalar rhs
  Tensor operator*(double other) const;
  // element wise scalar lhs
  friend Tensor operator*(double lhs, const Tensor &rhs);

  // dot product
  Tensor operator%(const Tensor &other) const;

  // subtract
  Tensor operator-(const Tensor &other) const;
  // subtract scalar rhs
  Tensor operator-(double other) const;
  // subtract scalar lhs
  friend Tensor operator-(double lhs, const Tensor &rhs);

  // auxilary operations
  Tensor operator-();
  Tensor exponential();
  Tensor max(double value);
  Tensor mean();
  Tensor transpose();
  Tensor reshape(std::vector<int> shape);

  Tensor sum();
  Tensor sum(std::vector<std::size_t> axes);

  [[nodiscard]] shapeView shape() const {
    return shapeView{heart->data.shape()};
  }

  // activation functions
  Tensor ReLU();
  Tensor ReLU_prime();
  Tensor Sigmoid();
  Tensor Sigmoid_prime();

private:
  // these can be turned into a struct which can then be made into a shared pointer
  mutable std::shared_ptr<Heart> heart;
};

#pragma once
#include <cstddef>
#include <vector>
#include <xtensor/core/xmath.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/io/xio.hpp>
class Tensor {

public:
  Tensor(xt::xarray<double> data,
         std::vector<Tensor> children = {},
         xt::xarray<double> gradients = {});

  // our __str__ method overload.
  friend std::ostream &operator<<(std::ostream &os, const Tensor &t) {
    os << "Tensor(" << t.data << ", " << "shape=" << t.shape() << ")\n";
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
  Tensor operator+(Tensor other);
  // addition scalar rhs
  Tensor operator+(double other);
  // addition scalar lhs
  friend Tensor operator+(double lhs, Tensor &rhs);

  // element wise
  Tensor operator*(Tensor other);
  // element wise scalar rhs
  Tensor operator*(double other);
  // element wise scalar lhs
  friend Tensor operator*(double lhs, Tensor &rhs);

  // dot product
  Tensor operator^(Tensor other);

  // subtract
  Tensor operator-(Tensor other);
  // subtract scalar rhs
  Tensor operator-(double other);
  // subtract scalar lhs
  friend Tensor operator-(double lhs, Tensor &rhs);

  // auxilary operations
  Tensor operator-();
  Tensor exponential();
  Tensor max(double value);
  Tensor mean();
  Tensor reshape(std::vector<int> &shape);
  shapeView shape() const {
    return shapeView{data.shape()};
  }

private:
  xt::xarray<double> data;
  xt::xarray<double> gradients;

  std::vector<Tensor> children;
  std::function<void()> backwards;

  // friends :)
  // +
  friend Tensor operator+(double, Tensor &);
  // -
  friend Tensor operator-(double, Tensor &);
  // *
  friend Tensor operator*(double, Tensor &);
  // dot and negate
  // friend Tensor operator^(Tensor &, Tensor &);
  // friend Tensor operator-(Tensor &);
};

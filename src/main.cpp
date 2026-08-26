#include <print>
#include <usul.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>

using namespace std;

int main() {
  xt::xarray<double> arr = {1.0, 2.0, 3.0};
  println("array: {}", arr.shape());

  xt::xarray<double> arr2 = {1.0, 2.0, 3.0};

  xt::xarray<double> result = arr + arr2;

  println("{}", result);

  return 0;
}

// Arguments: Doubles, Doubles
#include <stan/math/prim/prob/rayleigh_cdf_log.hpp>
#include <stan/math/prim/fun/exp.hpp>
#include <stan/math/prim/fun/log.hpp>

using stan::math::var;
using std::numeric_limits;
using std::vector;

class AgradCdfLogRayleigh : public AgradCdfLogTest {
 public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& cdf_log) {
    vector<double> param(2);

    param[0] = 1;  // y
    param[1] = 1;  // sigma
    parameters.push_back(param);
    cdf_log.push_back(std::log(0.393469340287367));  // expected cdf_log

    param[0] = 2;  // y
    param[1] = 1;  // sigma
    parameters.push_back(param);
    cdf_log.push_back(std::log(0.864664716763387));  // expected cdf_log

    param[0] = 3;  // y
    param[1] = 1;  // sigma
    parameters.push_back(param);
    cdf_log.push_back(std::log(0.988891003461758));  // expected cdf_log

    param[0] = 3.5;  // y
    param[1] = 7.2;  // sigma
    parameters.push_back(param);
    cdf_log.push_back(std::log(0.11143902440462770394));  // expected cdf_log
  }

  void invalid_values(vector<size_t>& index, vector<double>& value) {
    // y
    index.push_back(0U);
    value.push_back(-1.0);

    // sigma
    index.push_back(1U);
    value.push_back(0.0);

    index.push_back(1U);
    value.push_back(-1.0);

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());
  }

  bool has_lower_bound() { return true; }

  double lower_bound() { return 0.0; }

  bool has_upper_bound() { return false; }

  template <typename T_y, typename T_scale, typename T2, typename T3,
            typename T4, typename T5>
  stan::return_type_t<T_y, T_scale> cdf_log(const T_y& y, const T_scale& sigma,
                                            const T2&, const T3&, const T4&,
                                            const T5&) {
    return stan::math::rayleigh_cdf_log(y, sigma);
  }

  template <typename T_y, typename T_scale, typename T2, typename T3,
            typename T4, typename T5>
  stan::return_type_t<T_y, T_scale> cdf_log_function(const T_y& y,
                                                     const T_scale& sigma,
                                                     const T2&, const T3&,
                                                     const T4&, const T5&) {
    return log((1.0 - exp(-0.5 * y * y / (sigma * sigma))));
  }
};
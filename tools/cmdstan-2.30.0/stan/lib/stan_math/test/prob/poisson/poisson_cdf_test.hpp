// Arguments: Ints, Doubles
#include <stan/math/prim.hpp>

using stan::math::var;
using std::numeric_limits;
using std::vector;

class AgradCdfPoisson : public AgradCdfTest {
 public:
  void valid_values(vector<vector<double> >& parameters, vector<double>& cdf) {
    vector<double> param(2);

    param[0] = 17;    // n
    param[1] = 13.0;  // lambda
    parameters.push_back(param);
    cdf.push_back(0.8904649795242025600572);  // expected cdf

    param[0] = 82;    // n
    param[1] = 42.0;  // lambda
    parameters.push_back(param);
    cdf.push_back(0.9999999845303266798879);  // expected cdf

    param[0] = 0.0;  // n
    param[1] = 3.0;  // lambda
    parameters.push_back(param);
    cdf.push_back(0.04978706836786394446248);  // expected cdf
  }

  void invalid_values(vector<size_t>& index, vector<double>& value) {
    // lambda
    index.push_back(1U);
    value.push_back(-1e-5);

    index.push_back(1U);
    value.push_back(-1);
  }

  bool has_lower_bound() { return false; }

  bool has_upper_bound() { return false; }

  template <typename T_n, typename T_rate, typename T2, typename T3,
            typename T4, typename T5>
  stan::return_type_t<T_rate> cdf(const T_n& n, const T_rate& lambda, const T2&,
                                  const T3&, const T4&, const T5&) {
    return stan::math::poisson_cdf(n, lambda);
  }

  template <typename T_n, typename T_rate, typename T2, typename T3,
            typename T4, typename T5>
  stan::return_type_t<T_rate> cdf_function(const T_n& n, const T_rate& lambda,
                                           const T2&, const T3&, const T4&,
                                           const T5&) {
    using boost::math::lgamma;
    using stan::math::exp;
    using stan::math::lgamma;
    using stan::math::log;
    using stan::math::pow;

    stan::return_type_t<T_rate> cdf(0);
    for (int i = 0; i <= n; i++) {
      cdf += exp(i * log(lambda) - lgamma(i + 1));
    }
    cdf *= exp(-lambda);
    return cdf;
  }
};

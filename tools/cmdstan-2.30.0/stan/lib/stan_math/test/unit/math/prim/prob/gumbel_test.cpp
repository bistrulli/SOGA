#include <stan/math/prim.hpp>
#include <test/unit/math/prim/prob/vector_rng_test_helper.hpp>
#include <test/unit/math/prim/prob/util.hpp>
#include <boost/math/distributions.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

class GumbelTestRig : public VectorRealRNGTestRig {
 public:
  GumbelTestRig()
      : VectorRealRNGTestRig(
          10000, 10, {-2.5, -1.7, -0.1, 0.0, 2.0, 5.8}, {-3, -2, -1, 0, 2, 6},
          {}, {}, {0.1, 1.0, 2.5, 4.0}, {1, 2, 3, 4},
          {-1.0, -1.5, -2.5, -0.7, 0.0}, {-1, -2, -3, -4, 0}) {}

  template <typename T1, typename T2, typename T3, typename T_rng>
  auto generate_samples(const T1& mu, const T2& sigma, const T3& unused,
                        T_rng& rng) const {
    return stan::math::gumbel_rng(mu, sigma, rng);
  }

  std::vector<double> generate_quantiles(double mu, double sigma,
                                         double unused) const {
    std::vector<double> quantiles;
    double K = stan::math::round(2 * std::pow(N_, 0.4));
    boost::math::extreme_value_distribution<> dist(mu, sigma);

    for (int i = 1; i < K; ++i) {
      double frac = static_cast<double>(i) / K;
      quantiles.push_back(quantile(dist, frac));
    }
    quantiles.push_back(std::numeric_limits<double>::max());

    return quantiles;
  }
};

TEST(ProbDistributionsGumbel, errorCheck) {
  check_dist_throws_all_types(GumbelTestRig());
}

TEST(ProbDistributionsGumbel, distributionTest) {
  check_quantiles_real_real(GumbelTestRig());
}

TEST(ProbDistributionsGumbel, error_check) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::math::gumbel_rng(10.0, 2.0, rng));

  EXPECT_THROW(
      stan::math::gumbel_rng(stan::math::positive_infinity(), 2.0, rng),
      std::domain_error);
  EXPECT_THROW(stan::math::gumbel_rng(10.0, -2, rng), std::domain_error);
}

TEST(ProbDistributionsGumbel, chiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  int N = 10000;
  int K = stan::math::round(2 * std::pow(N, 0.4));

  std::vector<double> samples;
  for (int i = 0; i < N; ++i) {
    samples.push_back(stan::math::gumbel_rng(2.0, 1.0, rng));
  }

  // Generate quantiles from boost's Gumbel distribution
  boost::math::extreme_value_distribution<> dist(2.0, 1.0);
  std::vector<double> quantiles;
  for (int i = 1; i < K; ++i) {
    double frac = static_cast<double>(i) / K;
    quantiles.push_back(quantile(dist, frac));
  }
  quantiles.push_back(std::numeric_limits<double>::max());

  // Assert that they match
  assert_matches_quantiles(samples, quantiles, 1e-6);
}

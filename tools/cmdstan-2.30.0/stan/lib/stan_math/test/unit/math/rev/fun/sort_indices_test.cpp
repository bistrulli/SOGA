#include <stan/math/rev.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/fun/util.hpp>
#include <vector>

void test_sort_indices_asc(std::vector<double> val) {
  using stan::math::sort_indices_asc;

  std::vector<stan::math::var> x;
  for (size_t i = 0U; i < val.size(); i++)
    x.push_back(stan::math::var(val[i]));

  std::vector<int> val_sorted = sort_indices_asc(val);
  std::vector<int> x_sorted = sort_indices_asc(x);

  for (size_t i = 0U; i < val.size(); i++)
    EXPECT_EQ(val_sorted[i], x_sorted[i]);

  for (size_t i = 0U; i < val.size(); i++)
    for (size_t j = 0U; j < val.size(); j++)
      if (val_sorted[i] == val[j])
        EXPECT_EQ(x_sorted[i], x[j]);
      else
        EXPECT_FALSE(x_sorted[i] == x[j]);
}

void test_sort_indices_desc(std::vector<double> val) {
  using stan::math::sort_indices_desc;

  std::vector<stan::math::var> x;
  for (size_t i = 0U; i < val.size(); i++)
    x.push_back(stan::math::var(val[i]));

  std::vector<int> val_sorted = sort_indices_desc(val);
  std::vector<int> x_sorted = sort_indices_desc(x);

  for (size_t i = 0U; i < val.size(); i++)
    EXPECT_EQ(val_sorted[i], x_sorted[i]);

  for (size_t i = 0U; i < val.size(); i++)
    for (size_t j = 0U; j < val.size(); j++)
      if (val_sorted[i] == val[j])
        EXPECT_EQ(x_sorted[i], x[j]);
      else
        EXPECT_FALSE(x_sorted[i] == x[j]);
}

template <typename T, int R, int C>
void test_sort_indices_asc(Eigen::Matrix<T, R, C> val) {
  using stan::math::sort_indices_asc;

  const size_t val_size = val.size();

  std::vector<stan::math::var> x(val_size);
  for (size_t i = 0U; i < val_size; i++)
    x.data()[i] = stan::math::var(val[i]);

  std::vector<int> val_sorted = sort_indices_asc(val);
  std::vector<int> x_sorted = sort_indices_asc(x);

  for (size_t i = 0U; i < val_size; i++)
    EXPECT_EQ(val_sorted.data()[i], x_sorted.data()[i]);

  for (size_t i = 0U; i < val_size; i++)
    for (size_t j = 0U; j < val_size; j++)
      if (val_sorted.data()[i] == val.data()[j])
        EXPECT_EQ(x_sorted.data()[i], x.data()[j]);
      else
        EXPECT_FALSE(x_sorted.data()[i] == x.data()[j]);
}

template <typename T, int R, int C>
void test_sort_indices_desc(Eigen::Matrix<T, R, C> val) {
  using stan::math::sort_indices_desc;

  const size_t val_size = val.size();

  std::vector<stan::math::var> x(val_size);
  for (size_t i = 0U; i < val_size; i++)
    x.data()[i] = stan::math::var(val[i]);

  std::vector<int> val_sorted = sort_indices_desc(val);
  std::vector<int> x_sorted = sort_indices_desc(x);

  for (size_t i = 0U; i < val_size; i++)
    EXPECT_EQ(val_sorted.data()[i], x_sorted.data()[i]);

  for (size_t i = 0U; i < val_size; i++)
    for (size_t j = 0U; j < val_size; j++)
      if (val_sorted.data()[i] == val.data()[j])
        EXPECT_EQ(x_sorted.data()[i], x.data()[j]);
      else
        EXPECT_FALSE(x_sorted.data()[i] == x.data()[j]);
}

TEST(AgradRev, sort_indices) {
  std::vector<double> a;
  a.push_back(1);
  a.push_back(2);
  a.push_back(2);
  a.push_back(3);
  test_sort_indices_asc(a);
  test_sort_indices_desc(a);

  std::vector<double> b;
  b.push_back(1.1);
  b.push_back(2.2);
  b.push_back(33.1);
  b.push_back(-12.1);
  b.push_back(33.1);
  test_sort_indices_asc(b);
  test_sort_indices_desc(b);

  std::vector<double> c;
  c.push_back(1.1);
  c.push_back(-2);
  c.push_back(2.1);
  c.push_back(3);
  c.push_back(2.1);
  test_sort_indices_asc(c);
  test_sort_indices_desc(c);

  Eigen::RowVectorXd vec1(4);
  vec1 << 1, -33.1, 2.1, -33.1;
  test_sort_indices_asc(vec1);
  test_sort_indices_desc(vec1);

  Eigen::RowVectorXd vec2(5);
  vec2 << 1.1e-6, -2.3, 31.1, 1, -10.1;
  test_sort_indices_asc(vec2);
  test_sort_indices_desc(vec2);

  Eigen::VectorXd vec3(4);
  vec3 << -11.1, 2.2, -3.6, 2.2;
  test_sort_indices_asc(vec3);
  test_sort_indices_desc(vec3);

  Eigen::VectorXd vec4(3);
  vec4 << -10.1, 2.12, 3.102;
  test_sort_indices_asc(vec4);
  test_sort_indices_desc(vec4);

  Eigen::RowVectorXd vec5 = Eigen::RowVectorXd::Random(1, 10);
  test_sort_indices_asc(vec5);
  test_sort_indices_desc(vec5);

  Eigen::VectorXd vec6 = Eigen::VectorXd::Random(20, 1);
  test_sort_indices_asc(vec6);
  test_sort_indices_desc(vec6);
}

TEST(AgradRev, sort_indices_no_thrown) {
  using stan::math::sort_indices_asc;
  using stan::math::sort_indices_desc;

  std::vector<stan::math::var> vec0;
  EXPECT_EQ(0U, vec0.size());
  EXPECT_NO_THROW(sort_indices_asc(vec0));
  EXPECT_NO_THROW(sort_indices_desc(vec0));

  Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> vec1;
  EXPECT_EQ(0, vec1.size());
  EXPECT_NO_THROW(sort_indices_asc(vec1));
  EXPECT_NO_THROW(sort_indices_desc(vec1));

  Eigen::Matrix<stan::math::var, 1, Eigen::Dynamic> vec2;
  EXPECT_EQ(0, vec2.size());
  EXPECT_NO_THROW(sort_indices_asc(vec2));
  EXPECT_NO_THROW(sort_indices_desc(vec2));
}

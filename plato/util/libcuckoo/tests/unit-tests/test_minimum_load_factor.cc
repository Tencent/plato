#include <iostream>

#include <catch.hpp>

#include "unit_test_util.hh"
#include <libcuckoo/cuckoohash_config.hh>
#include <libcuckoo/cuckoohash_map.hh>

TEST_CASE("minimum load factor initialized to default",
          "[minimum load factor]") {
  IntIntTable tbl;
  REQUIRE(tbl.minimum_load_factor() == LIBCUCKOO_DEFAULT_MINIMUM_LOAD_FACTOR);
}

class BadHashFunction {
public:
  size_t operator()(int) { return 0; }
};

TEST_CASE("caps automatic expansion", "[minimum load fator]") {
  const size_t slot_per_bucket = 4;
  cuckoohash_map<int, int, BadHashFunction, std::equal_to<int>,
                 std::allocator<std::pair<const int, int>>, slot_per_bucket>
      tbl(16);
  tbl.minimum_load_factor(0.6);

  for (size_t i = 0; i < 2 * slot_per_bucket; ++i) {
    tbl.insert(i, i);
  }

  REQUIRE_THROWS_AS(tbl.insert(2 * slot_per_bucket, 0),
                    libcuckoo_load_factor_too_low);
}

TEST_CASE("invalid minimum load factor", "[minimum load factor]") {
  IntIntTable tbl;
  REQUIRE_THROWS_AS(tbl.minimum_load_factor(-0.01), std::invalid_argument);
  REQUIRE_THROWS_AS(tbl.minimum_load_factor(1.01), std::invalid_argument);
}

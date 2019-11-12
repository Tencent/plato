#include <iostream>

#include <catch.hpp>

#include "unit_test_util.hh"
#include <libcuckoo/cuckoohash_config.hh>
#include <libcuckoo/cuckoohash_map.hh>

TEST_CASE("maximum hashpower initialized to default", "[maximum hash power]") {
  IntIntTable tbl;
  REQUIRE(tbl.maximum_hashpower() == LIBCUCKOO_NO_MAXIMUM_HASHPOWER);
}

TEST_CASE("caps any expansion", "[maximum hash power]") {
  IntIntTable tbl(1);
  tbl.maximum_hashpower(1);
  for (size_t i = 0; i < 2 * tbl.slot_per_bucket(); ++i) {
    tbl.insert(i, i);
  }

  REQUIRE(tbl.hashpower() == 1);
  REQUIRE_THROWS_AS(tbl.insert(2 * tbl.slot_per_bucket(), 0),
                    libcuckoo_maximum_hashpower_exceeded);
  REQUIRE_THROWS_AS(tbl.rehash(2), libcuckoo_maximum_hashpower_exceeded);
  REQUIRE_THROWS_AS(tbl.reserve(4 * tbl.slot_per_bucket()),
                    libcuckoo_maximum_hashpower_exceeded);
}

TEST_CASE("no maximum hash power", "[maximum hash power]") {
  // It's difficult to check that we actually don't ever set a maximum hash
  // power, but if we explicitly unset it, we should be able to expand beyond
  // the limit that we had previously set.
  IntIntTable tbl(1);
  tbl.maximum_hashpower(1);
  REQUIRE_THROWS_AS(tbl.rehash(2), libcuckoo_maximum_hashpower_exceeded);

  tbl.maximum_hashpower(2);
  tbl.rehash(2);
  REQUIRE(tbl.hashpower() == 2);
  REQUIRE_THROWS_AS(tbl.rehash(3), libcuckoo_maximum_hashpower_exceeded);

  tbl.maximum_hashpower(LIBCUCKOO_NO_MAXIMUM_HASHPOWER);
  tbl.rehash(10);
  REQUIRE(tbl.hashpower() == 10);
}

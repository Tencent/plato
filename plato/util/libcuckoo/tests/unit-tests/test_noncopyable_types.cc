#include <catch.hpp>

#include <string>
#include <utility>

#include "unit_test_util.hh"
#include <libcuckoo/cuckoohash_map.hh>

using Tbl = UniquePtrTable<int>;
using Uptr = std::unique_ptr<int>;

const size_t TBL_INIT = 1;
const size_t TBL_SIZE = TBL_INIT * Tbl::slot_per_bucket() * 2;

void check_key_eq(Tbl &tbl, int key, int expected_val) {
  REQUIRE(tbl.contains(Uptr(new int(key))));
  tbl.find_fn(Uptr(new int(key)), [expected_val](const Uptr &ptr) {
    REQUIRE(*ptr == expected_val);
  });
}

TEST_CASE("noncopyable insert and update", "[noncopyable]") {
  Tbl tbl(TBL_INIT);
  for (size_t i = 0; i < TBL_SIZE; ++i) {
    REQUIRE(tbl.insert(Uptr(new int(i)), Uptr(new int(i))));
  }
  for (size_t i = 0; i < TBL_SIZE; ++i) {
    check_key_eq(tbl, i, i);
  }
  for (size_t i = 0; i < TBL_SIZE; ++i) {
    tbl.update(Uptr(new int(i)), Uptr(new int(i + 1)));
  }
  for (size_t i = 0; i < TBL_SIZE; ++i) {
    check_key_eq(tbl, i, i + 1);
  }
}

TEST_CASE("noncopyable upsert", "[noncopyable]") {
  Tbl tbl(TBL_INIT);
  auto increment = [](Uptr &ptr) { *ptr += 1; };
  for (size_t i = 0; i < TBL_SIZE; ++i) {
    tbl.upsert(Uptr(new int(i)), increment, Uptr(new int(i)));
  }
  for (size_t i = 0; i < TBL_SIZE; ++i) {
    check_key_eq(tbl, i, i);
  }
  for (size_t i = 0; i < TBL_SIZE; ++i) {
    tbl.upsert(Uptr(new int(i)), increment, Uptr(new int(i)));
  }
  for (size_t i = 0; i < TBL_SIZE; ++i) {
    check_key_eq(tbl, i, i + 1);
  }
}

TEST_CASE("noncopyable iteration", "[noncopyable]") {
  Tbl tbl(TBL_INIT);
  for (size_t i = 0; i < TBL_SIZE; ++i) {
    tbl.insert(Uptr(new int(i)), Uptr(new int(i)));
  }
  {
    auto locked_tbl = tbl.lock_table();
    for (auto &kv : locked_tbl) {
      REQUIRE(*kv.first == *kv.second);
      *kv.second += 1;
    }
  }
  {
    auto locked_tbl = tbl.lock_table();
    for (auto &kv : locked_tbl) {
      REQUIRE(*kv.first == *kv.second - 1);
    }
  }
}

TEST_CASE("nested table", "[noncopyable]") {
  typedef cuckoohash_map<char, std::string> inner_tbl;
  typedef cuckoohash_map<std::string, std::unique_ptr<inner_tbl>> nested_tbl;
  nested_tbl tbl;
  std::string keys[] = {"abc", "def"};
  for (std::string &k : keys) {
    tbl.insert(std::string(k), nested_tbl::mapped_type(new inner_tbl));
    tbl.update_fn(k, [&k](nested_tbl::mapped_type &t) {
      for (char c : k) {
        t->insert(c, std::string(k));
      }
    });
  }
  for (std::string &k : keys) {
    REQUIRE(tbl.contains(k));
    tbl.update_fn(k, [&k](nested_tbl::mapped_type &t) {
      for (char c : k) {
        REQUIRE(t->find(c) == k);
      }
    });
  }
}

TEST_CASE("noncopyable insert lifetime") {
  Tbl tbl;

  // Successful insert
  SECTION("Successful insert") {
    Uptr key(new int(20));
    Uptr value(new int(20));
    REQUIRE(tbl.insert(std::move(key), std::move(value)));
    REQUIRE(!static_cast<bool>(key));
    REQUIRE(!static_cast<bool>(value));
  }

  // Unsuccessful insert
  SECTION("Unsuccessful insert") {
    tbl.insert(new int(20), new int(20));
    Uptr key(new int(20));
    Uptr value(new int(30));
    REQUIRE_FALSE(tbl.insert(std::move(key), std::move(value)));
    REQUIRE(static_cast<bool>(key));
    REQUIRE(static_cast<bool>(value));
  }
}

TEST_CASE("noncopyable erase_fn") {
  Tbl tbl;
  tbl.insert(new int(10), new int(10));
  auto decrement_and_erase = [](Uptr &p) {
    --(*p);
    return *p == 0;
  };
  Uptr k(new int(10));
  for (int i = 0; i < 9; ++i) {
    tbl.erase_fn(k, decrement_and_erase);
    REQUIRE(tbl.contains(k));
  }
  tbl.erase_fn(k, decrement_and_erase);
  REQUIRE_FALSE(tbl.contains(k));
}

TEST_CASE("noncopyable uprase_fn") {
  Tbl tbl;
  auto decrement_and_erase = [](Uptr &p) {
    --(*p);
    return *p == 0;
  };
  REQUIRE(
      tbl.uprase_fn(Uptr(new int(10)), decrement_and_erase, Uptr(new int(10))));
  Uptr k(new int(10)), v(new int(10));
  for (int i = 0; i < 10; ++i) {
    REQUIRE_FALSE(
        tbl.uprase_fn(std::move(k), decrement_and_erase, std::move(v)));
    REQUIRE((k && v));
    if (i < 9) {
      REQUIRE(tbl.contains(k));
    } else {
      REQUIRE_FALSE(tbl.contains(k));
    }
  }
}

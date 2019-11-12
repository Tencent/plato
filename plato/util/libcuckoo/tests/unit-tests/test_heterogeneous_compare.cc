#include <catch.hpp>

#include <libcuckoo/cuckoohash_map.hh>

size_t int_constructions;
size_t copy_constructions;
size_t destructions;
size_t foo_comparisons;
size_t int_comparisons;
size_t foo_hashes;
size_t int_hashes;

class Foo {
public:
  int val;

  Foo(int v) {
    ++int_constructions;
    val = v;
  }

  Foo(const Foo &x) {
    ++copy_constructions;
    val = x.val;
  }

  ~Foo() { ++destructions; }
};

class foo_eq {
public:
  bool operator()(const Foo &left, const Foo &right) const {
    ++foo_comparisons;
    return left.val == right.val;
  }

  bool operator()(const Foo &left, const int right) const {
    ++int_comparisons;
    return left.val == right;
  }
};

class foo_hasher {
public:
  size_t operator()(const Foo &x) const {
    ++foo_hashes;
    return static_cast<size_t>(x.val);
  }

  size_t operator()(const int x) const {
    ++int_hashes;
    return static_cast<size_t>(x);
  }
};

typedef cuckoohash_map<Foo, bool, foo_hasher, foo_eq> foo_map;

TEST_CASE("heterogeneous compare", "[heterogeneous compare]") {
  // setup code
  int_constructions = 0;
  copy_constructions = 0;
  destructions = 0;
  foo_comparisons = 0;
  int_comparisons = 0;
  foo_hashes = 0;
  int_hashes = 0;

  SECTION("insert") {
    {
      foo_map map;
      map.insert(0, true);
    }
    REQUIRE(int_constructions == 1);
    REQUIRE(copy_constructions == 0);
    REQUIRE(destructions == 1);
    REQUIRE(foo_comparisons == 0);
    REQUIRE(int_comparisons == 0);
    REQUIRE(foo_hashes == 0);
    REQUIRE(int_hashes == 1);
  }

  SECTION("foo insert") {
    {
      foo_map map;
      map.insert(Foo(0), true);
    }
    REQUIRE(int_constructions == 1);
    REQUIRE(copy_constructions == 1);
    // One destruction of passed-in and moved argument, and one after the
    // table is destroyed.
    REQUIRE(destructions == 2);
    REQUIRE(foo_comparisons == 0);
    REQUIRE(int_comparisons == 0);
    REQUIRE(foo_hashes == 1);
    REQUIRE(int_hashes == 0);
  }

  SECTION("insert_or_assign") {
    {
      foo_map map;
      map.insert_or_assign(0, true);
      map.insert_or_assign(0, false);
      REQUIRE_FALSE(map.find(0));
    }
    REQUIRE(int_constructions == 1);
    REQUIRE(copy_constructions == 0);
    REQUIRE(destructions == 1);
    REQUIRE(foo_comparisons == 0);
    REQUIRE(int_comparisons == 2);
    REQUIRE(foo_hashes == 0);
    REQUIRE(int_hashes == 3);
  }

  SECTION("foo insert_or_assign") {
    {
      foo_map map;
      map.insert_or_assign(Foo(0), true);
      map.insert_or_assign(Foo(0), false);
      REQUIRE_FALSE(map.find(Foo(0)));
    }
    REQUIRE(int_constructions == 3);
    REQUIRE(copy_constructions == 1);
    // Three destructions of Foo arguments, and one in table destruction
    REQUIRE(destructions == 4);
    REQUIRE(foo_comparisons == 2);
    REQUIRE(int_comparisons == 0);
    REQUIRE(foo_hashes == 3);
    REQUIRE(int_hashes == 0);
  }

  SECTION("find") {
    {
      foo_map map;
      map.insert(0, true);
      bool val;
      map.find(0, val);
      REQUIRE(val);
      REQUIRE(map.find(0, val) == true);
      REQUIRE(map.find(1, val) == false);
    }
    REQUIRE(int_constructions == 1);
    REQUIRE(copy_constructions == 0);
    REQUIRE(destructions == 1);
    REQUIRE(foo_comparisons == 0);
    REQUIRE(int_comparisons == 2);
    REQUIRE(foo_hashes == 0);
    REQUIRE(int_hashes == 4);
  }

  SECTION("foo find") {
    {
      foo_map map;
      map.insert(0, true);
      bool val;
      map.find(Foo(0), val);
      REQUIRE(val);
      REQUIRE(map.find(Foo(0), val) == true);
      REQUIRE(map.find(Foo(1), val) == false);
    }
    REQUIRE(int_constructions == 4);
    REQUIRE(copy_constructions == 0);
    REQUIRE(destructions == 4);
    REQUIRE(foo_comparisons == 2);
    REQUIRE(int_comparisons == 0);
    REQUIRE(foo_hashes == 3);
    REQUIRE(int_hashes == 1);
  }

  SECTION("contains") {
    {
      foo_map map(0);
      map.rehash(2);
      map.insert(0, true);
      REQUIRE(map.contains(0));
      // Shouldn't do comparison because of different partial key
      REQUIRE(!map.contains(4));
    }
    REQUIRE(int_constructions == 1);
    REQUIRE(copy_constructions == 0);
    REQUIRE(destructions == 1);
    REQUIRE(foo_comparisons == 0);
    REQUIRE(int_comparisons == 1);
    REQUIRE(foo_hashes == 0);
    REQUIRE(int_hashes == 3);
  }

  SECTION("erase") {
    {
      foo_map map;
      map.insert(0, true);
      REQUIRE(map.erase(0));
      REQUIRE(!map.contains(0));
    }
    REQUIRE(int_constructions == 1);
    REQUIRE(copy_constructions == 0);
    REQUIRE(destructions == 1);
    REQUIRE(foo_comparisons == 0);
    REQUIRE(int_comparisons == 1);
    REQUIRE(foo_hashes == 0);
    REQUIRE(int_hashes == 3);
  }

  SECTION("update") {
    {
      foo_map map;
      map.insert(0, true);
      REQUIRE(map.update(0, false));
      REQUIRE(!map.find(0));
    }
    REQUIRE(int_constructions == 1);
    REQUIRE(copy_constructions == 0);
    REQUIRE(destructions == 1);
    REQUIRE(foo_comparisons == 0);
    REQUIRE(int_comparisons == 2);
    REQUIRE(foo_hashes == 0);
    REQUIRE(int_hashes == 3);
  }

  SECTION("update_fn") {
    {
      foo_map map;
      map.insert(0, true);
      REQUIRE(map.update_fn(0, [](bool &val) { val = !val; }));
      REQUIRE(!map.find(0));
    }
    REQUIRE(int_constructions == 1);
    REQUIRE(copy_constructions == 0);
    REQUIRE(destructions == 1);
    REQUIRE(foo_comparisons == 0);
    REQUIRE(int_comparisons == 2);
    REQUIRE(foo_hashes == 0);
    REQUIRE(int_hashes == 3);
  }

  SECTION("upsert") {
    {
      foo_map map(0);
      map.rehash(2);
      auto neg = [](bool &val) { val = !val; };
      map.upsert(0, neg, true);
      map.upsert(0, neg, true);
      // Shouldn't do comparison because of different partial key
      map.upsert(4, neg, false);
      REQUIRE(!map.find(0));
      REQUIRE(!map.find(4));
    }
    REQUIRE(int_constructions == 2);
    REQUIRE(copy_constructions == 0);
    REQUIRE(destructions == 2);
    REQUIRE(foo_comparisons == 0);
    REQUIRE(int_comparisons == 3);
    REQUIRE(foo_hashes == 0);
    REQUIRE(int_hashes == 5);
  }

  SECTION("uprase_fn") {
    {
      foo_map map(0);
      map.rehash(2);
      auto fn = [](bool &val) {
        val = !val;
        return val;
      };
      REQUIRE(map.uprase_fn(0, fn, true));
      REQUIRE_FALSE(map.uprase_fn(0, fn, true));
      REQUIRE(map.contains(0));
      REQUIRE_FALSE(map.uprase_fn(0, fn, true));
      REQUIRE_FALSE(map.contains(0));
    }
    REQUIRE(int_constructions == 1);
    REQUIRE(copy_constructions == 0);
    REQUIRE(destructions == 1);
    REQUIRE(foo_comparisons == 0);
    REQUIRE(int_comparisons == 3);
    REQUIRE(foo_hashes == 0);
    REQUIRE(int_hashes == 5);
  }
}

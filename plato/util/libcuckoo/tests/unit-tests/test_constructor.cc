#include <catch.hpp>

#include <array>
#include <cmath>
#include <stdexcept>

#include "unit_test_util.hh"
#include <libcuckoo/cuckoohash_map.hh>

TEST_CASE("default size", "[constructor]") {
  IntIntTable tbl;
  REQUIRE(tbl.size() == 0);
  REQUIRE(tbl.empty());
  if (LIBCUCKOO_DEFAULT_SIZE < 4) {
    REQUIRE(tbl.hashpower() == 0);
  } else {
    REQUIRE(tbl.hashpower() == (size_t)log2(LIBCUCKOO_DEFAULT_SIZE / 4));
  }
  REQUIRE(tbl.bucket_count() == 1UL << tbl.hashpower());
  REQUIRE(tbl.load_factor() == 0);
}

TEST_CASE("given size", "[constructor]") {
  IntIntTable tbl(1);
  REQUIRE(tbl.size() == 0);
  REQUIRE(tbl.empty());
  REQUIRE(tbl.hashpower() == 0);
  REQUIRE(tbl.bucket_count() == 1);
  REQUIRE(tbl.load_factor() == 0);
}

TEST_CASE("frees even with exceptions", "[constructor]") {
  typedef IntIntTableWithAlloc<TrackingAllocator<int, 0>> no_space_table;
  // Should throw when allocating anything
  REQUIRE_THROWS_AS(no_space_table(1), std::bad_alloc);
  REQUIRE(get_unfreed_bytes() == 0);

  typedef IntIntTableWithAlloc<
      TrackingAllocator<int, UnitTestInternalAccess::IntIntBucketSize * 2>>
      some_space_table;
  // Should throw when allocating things after the bucket
  REQUIRE_THROWS_AS(some_space_table(1), std::bad_alloc);
  REQUIRE(get_unfreed_bytes() == 0);
}

struct StatefulHash {
  StatefulHash(int state_) : state(state_) {}
  size_t operator()(int x) const { return x; }
  int state;
};

struct StatefulKeyEqual {
  StatefulKeyEqual(int state_) : state(state_) {}
  bool operator()(int x, int y) const { return x == y; }
  int state;
};

template <typename T> struct StatefulAllocator {
  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;
  using size_type = size_t;
  using difference_type = ptrdiff_t;

  template <typename U> struct rebind { using other = StatefulAllocator<U>; };

  StatefulAllocator() : state(0) {}

  StatefulAllocator(int state_) : state(state_) {}

  template <typename U>
  StatefulAllocator(const StatefulAllocator<U> &other) : state(other.state) {}

  T *allocate(size_t n) { return std::allocator<T>().allocate(n); }

  void deallocate(T *p, size_t n) { std::allocator<T>().deallocate(p, n); }

  template <typename U, class... Args> void construct(U *p, Args &&... args) {
    new ((void *)p) U(std::forward<Args>(args)...);
  }

  template <typename U> void destroy(U *p) { p->~U(); }

  StatefulAllocator select_on_container_copy_construction() const {
    return StatefulAllocator();
  }

  using propagate_on_container_swap = std::integral_constant<bool, true>;

  int state;
};

template <typename T, typename U>
bool operator==(const StatefulAllocator<T> &a1,
                const StatefulAllocator<U> &a2) {
  return a1.state == a2.state;
}

template <typename T, typename U>
bool operator!=(const StatefulAllocator<T> &a1,
                const StatefulAllocator<U> &a2) {
  return a1.state != a2.state;
}

using alloc_t = StatefulAllocator<std::pair<const int, int>>;
using tbl_t =
    cuckoohash_map<int, int, StatefulHash, StatefulKeyEqual, alloc_t, 4>;

TEST_CASE("stateful components", "[constructor]") {
  tbl_t map(8, StatefulHash(10), StatefulKeyEqual(20), alloc_t(30));
  REQUIRE(map.hash_function().state == 10);
  for (int i = 0; i < 100; ++i) {
    REQUIRE(map.hash_function()(i) == i);
  }
  REQUIRE(map.key_eq().state == 20);
  for (int i = 0; i < 100; ++i) {
    REQUIRE(map.key_eq()(i, i));
    REQUIRE_FALSE(map.key_eq()(i, i + 1));
  }
  REQUIRE(map.get_allocator().state == 30);
}

TEST_CASE("range constructor", "[constructor]") {
  std::array<typename alloc_t::value_type, 3> elems{{{1, 2}, {3, 4}, {5, 6}}};
  tbl_t map(elems.begin(), elems.end(), 3, StatefulHash(10),
            StatefulKeyEqual(20), alloc_t(30));
  REQUIRE(map.hash_function().state == 10);
  REQUIRE(map.key_eq().state == 20);
  REQUIRE(map.get_allocator().state == 30);
  for (int i = 1; i <= 5; i += 2) {
    REQUIRE(map.find(i) == i + 1);
  }
}

TEST_CASE("copy constructor", "[constructor]") {
  tbl_t map(0, StatefulHash(10), StatefulKeyEqual(20), alloc_t(30));
  REQUIRE(map.get_allocator().state == 30);
  tbl_t map2(map);
  REQUIRE(map2.hash_function().state == 10);
  REQUIRE(map2.key_eq().state == 20);
  REQUIRE(map2.get_allocator().state == 0);
}

TEST_CASE("copy constructor other allocator", "[constructor]") {
  tbl_t map(0, StatefulHash(10), StatefulKeyEqual(20), alloc_t(30));
  tbl_t map2(map, map.get_allocator());
  REQUIRE(map2.hash_function().state == 10);
  REQUIRE(map2.key_eq().state == 20);
  REQUIRE(map2.get_allocator().state == 30);
}

TEST_CASE("move constructor", "[constructor]") {
  tbl_t map(10, StatefulHash(10), StatefulKeyEqual(20), alloc_t(30));
  map.insert(10, 10);
  tbl_t map2(std::move(map));
  REQUIRE(map.size() == 0);
  REQUIRE(map2.size() == 1);
  REQUIRE(map2.hash_function().state == 10);
  REQUIRE(map2.key_eq().state == 20);
  REQUIRE(map2.get_allocator().state == 30);
}

TEST_CASE("move constructor different allocator", "[constructor]") {
  tbl_t map(10, StatefulHash(10), StatefulKeyEqual(20), alloc_t(30));
  map.insert(10, 10);
  tbl_t map2(std::move(map), alloc_t(40));
  REQUIRE(map.size() == 1);
  REQUIRE(map.hash_function().state == 10);
  REQUIRE(map.key_eq().state == 20);
  REQUIRE(map.get_allocator().state == 30);

  REQUIRE(map2.size() == 1);
  REQUIRE(map2.hash_function().state == 10);
  REQUIRE(map2.key_eq().state == 20);
  REQUIRE(map2.get_allocator().state == 40);
}

TEST_CASE("initializer list constructor", "[constructor]") {
  tbl_t map({{1, 2}, {3, 4}, {5, 6}}, 3, StatefulHash(10), StatefulKeyEqual(20),
            alloc_t(30));
  REQUIRE(map.hash_function().state == 10);
  REQUIRE(map.key_eq().state == 20);
  REQUIRE(map.get_allocator().state == 30);
  for (int i = 1; i <= 5; i += 2) {
    REQUIRE(map.find(i) == i + 1);
  }
}

TEST_CASE("swap maps", "[constructor]") {
  tbl_t map({{1, 2}}, 1, StatefulHash(10), StatefulKeyEqual(20), alloc_t(30));
  tbl_t map2({{3, 4}}, 1, StatefulHash(40), StatefulKeyEqual(50), alloc_t(60));
  map.swap(map2);

  REQUIRE(map.size() == 1);
  REQUIRE(map.hash_function().state == 40);
  REQUIRE(map.key_eq().state == 50);
  REQUIRE(map.get_allocator().state == 60);

  REQUIRE(map2.size() == 1);
  REQUIRE(map2.hash_function().state == 10);
  REQUIRE(map2.key_eq().state == 20);
  REQUIRE(map2.get_allocator().state == 30);

  std::swap(map, map2);

  REQUIRE(map.size() == 1);
  REQUIRE(map.hash_function().state == 10);
  REQUIRE(map.key_eq().state == 20);
  REQUIRE(map.get_allocator().state == 30);

  REQUIRE(map2.size() == 1);
  REQUIRE(map2.hash_function().state == 40);
  REQUIRE(map2.key_eq().state == 50);
  REQUIRE(map2.get_allocator().state == 60);
}

TEST_CASE("copy assign different allocators", "[constructor]") {
  tbl_t map({{1, 2}}, 1, StatefulHash(10), StatefulKeyEqual(20), alloc_t(30));
  tbl_t map2({{3, 4}}, 1, StatefulHash(40), StatefulKeyEqual(50), alloc_t(60));

  map = map2;
  REQUIRE(map.size() == 1);
  REQUIRE(map.find(3) == 4);
  REQUIRE(map.hash_function().state == 40);
  REQUIRE(map.key_eq().state == 50);
  REQUIRE(map.get_allocator().state == 30);

  REQUIRE(map2.size() == 1);
  REQUIRE(map2.hash_function().state == 40);
  REQUIRE(map2.key_eq().state == 50);
  REQUIRE(map2.get_allocator().state == 60);
}

TEST_CASE("move assign different allocators", "[constructor]") {
  tbl_t map({{1, 2}}, 1, StatefulHash(10), StatefulKeyEqual(20), alloc_t(30));
  tbl_t map2({{3, 4}}, 1, StatefulHash(40), StatefulKeyEqual(50), alloc_t(60));

  map = std::move(map2);
  REQUIRE(map.size() == 1);
  REQUIRE(map.find(3) == 4);
  REQUIRE(map.hash_function().state == 40);
  REQUIRE(map.key_eq().state == 50);
  REQUIRE(map.get_allocator().state == 30);

  REQUIRE(map2.hash_function().state == 40);
  REQUIRE(map2.key_eq().state == 50);
  REQUIRE(map2.get_allocator().state == 60);
}

TEST_CASE("move assign same allocators", "[constructor]") {
  tbl_t map({{1, 2}}, 1, StatefulHash(10), StatefulKeyEqual(20), alloc_t(30));
  tbl_t map2({{3, 4}}, 1, StatefulHash(40), StatefulKeyEqual(50), alloc_t(30));

  map = std::move(map2);
  REQUIRE(map.size() == 1);
  REQUIRE(map.find(3) == 4);
  REQUIRE(map.hash_function().state == 40);
  REQUIRE(map.key_eq().state == 50);
  REQUIRE(map.get_allocator().state == 30);

  REQUIRE(map2.size() == 0);
  REQUIRE(map2.hash_function().state == 40);
  REQUIRE(map2.key_eq().state == 50);
  REQUIRE(map2.get_allocator().state == 30);
}

TEST_CASE("initializer list assignment", "[constructor]") {
  tbl_t map({{1, 2}}, 1, StatefulHash(10), StatefulKeyEqual(20), alloc_t(30));
  REQUIRE(map.find(1) == 2);
  map = {{3, 4}};
  REQUIRE(map.find(3) == 4);
}

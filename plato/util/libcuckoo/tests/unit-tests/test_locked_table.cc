#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include <catch.hpp>

#include "unit_test_util.hh"
#include <libcuckoo/cuckoohash_map.hh>

TEST_CASE("locked_table typedefs", "[locked_table]") {
  using Tbl = IntIntTable;
  using Ltbl = Tbl::locked_table;
  const bool key_type = std::is_same<Tbl::key_type, Ltbl::key_type>::value;
  const bool mapped_type =
      std::is_same<Tbl::mapped_type, Ltbl::mapped_type>::value;
  const bool value_type =
      std::is_same<Tbl::value_type, Ltbl::value_type>::value;
  const bool size_type = std::is_same<Tbl::size_type, Ltbl::size_type>::value;
  const bool difference_type =
      std::is_same<Tbl::difference_type, Ltbl::difference_type>::value;
  const bool hasher = std::is_same<Tbl::hasher, Ltbl::hasher>::value;
  const bool key_equal = std::is_same<Tbl::key_equal, Ltbl::key_equal>::value;
  const bool allocator_type =
      std::is_same<Tbl::allocator_type, Ltbl::allocator_type>::value;
  const bool reference = std::is_same<Tbl::reference, Ltbl::reference>::value;
  const bool const_reference =
      std::is_same<Tbl::const_reference, Ltbl::const_reference>::value;
  const bool pointer = std::is_same<Tbl::pointer, Ltbl::pointer>::value;
  const bool const_pointer =
      std::is_same<Tbl::const_pointer, Ltbl::const_pointer>::value;
  REQUIRE(key_type);
  REQUIRE(mapped_type);
  REQUIRE(value_type);
  REQUIRE(size_type);
  REQUIRE(difference_type);
  REQUIRE(hasher);
  REQUIRE(key_equal);
  REQUIRE(allocator_type);
  REQUIRE(reference);
  REQUIRE(const_reference);
  REQUIRE(pointer);
  REQUIRE(const_pointer);
}

TEST_CASE("locked_table move", "[locked_table]") {
  IntIntTable tbl;

  SECTION("move constructor") {
    auto lt = tbl.lock_table();
    auto lt2(std::move(lt));
    REQUIRE(!lt.is_active());
    REQUIRE(lt2.is_active());
  }

  SECTION("move assignment") {
    auto lt = tbl.lock_table();
    auto lt2 = std::move(lt);
    REQUIRE(!lt.is_active());
    REQUIRE(lt2.is_active());
  }

  SECTION("iterators compare after table is moved") {
    auto lt1 = tbl.lock_table();
    auto it1 = lt1.begin();
    auto it2 = lt1.begin();
    REQUIRE(it1 == it2);
    auto lt2(std::move(lt1));
    REQUIRE(it1 == it2);
  }
}

TEST_CASE("locked_table unlock", "[locked_table]") {
  IntIntTable tbl;
  tbl.insert(10, 10);
  auto lt = tbl.lock_table();
  lt.unlock();
  REQUIRE(!lt.is_active());
}

TEST_CASE("locked_table info", "[locked_table]") {
  IntIntTable tbl;
  tbl.insert(10, 10);
  auto lt = tbl.lock_table();
  REQUIRE(lt.is_active());

  // We should still be able to call table info operations on the
  // cuckoohash_map instance, because they shouldn't take locks.

  REQUIRE(lt.slot_per_bucket() == tbl.slot_per_bucket());
  REQUIRE(lt.get_allocator() == tbl.get_allocator());
  REQUIRE(lt.hashpower() == tbl.hashpower());
  REQUIRE(lt.bucket_count() == tbl.bucket_count());
  REQUIRE(lt.empty() == tbl.empty());
  REQUIRE(lt.size() == tbl.size());
  REQUIRE(lt.capacity() == tbl.capacity());
  REQUIRE(lt.load_factor() == tbl.load_factor());
  REQUIRE_THROWS_AS(lt.minimum_load_factor(1.01), std::invalid_argument);
  lt.minimum_load_factor(lt.minimum_load_factor() * 2);
  lt.rehash(5);
  REQUIRE_THROWS_AS(lt.maximum_hashpower(lt.hashpower() - 1),
                    std::invalid_argument);
  lt.maximum_hashpower(lt.hashpower() + 1);
  REQUIRE(lt.maximum_hashpower() == tbl.maximum_hashpower());
}

TEST_CASE("locked_table clear", "[locked_table]") {
  IntIntTable tbl;
  tbl.insert(10, 10);
  auto lt = tbl.lock_table();
  REQUIRE(lt.size() == 1);
  lt.clear();
  REQUIRE(lt.size() == 0);
  lt.clear();
  REQUIRE(lt.size() == 0);
}

TEST_CASE("locked_table insert duplicate", "[locked_table]") {
  IntIntTable tbl;
  tbl.insert(10, 10);
  {
    auto lt = tbl.lock_table();
    auto result = lt.insert(10, 20);
    REQUIRE(result.first->first == 10);
    REQUIRE(result.first->second == 10);
    REQUIRE_FALSE(result.second);
    result.first->second = 50;
  }
  REQUIRE(tbl.find(10) == 50);
}

TEST_CASE("locked_table insert new key", "[locked_table]") {
  IntIntTable tbl;
  tbl.insert(10, 10);
  {
    auto lt = tbl.lock_table();
    auto result = lt.insert(20, 20);
    REQUIRE(result.first->first == 20);
    REQUIRE(result.first->second == 20);
    REQUIRE(result.second);
    result.first->second = 50;
  }
  REQUIRE(tbl.find(10) == 10);
  REQUIRE(tbl.find(20) == 50);
}

TEST_CASE("locked_table insert lifetime", "[locked_table]") {
  UniquePtrTable<int> tbl;

  SECTION("Successful insert") {
    auto lt = tbl.lock_table();
    std::unique_ptr<int> key(new int(20));
    std::unique_ptr<int> value(new int(20));
    auto result = lt.insert(std::move(key), std::move(value));
    REQUIRE(*result.first->first == 20);
    REQUIRE(*result.first->second == 20);
    REQUIRE(result.second);
    REQUIRE(!static_cast<bool>(key));
    REQUIRE(!static_cast<bool>(value));
  }

  SECTION("Unsuccessful insert") {
    tbl.insert(new int(20), new int(20));
    auto lt = tbl.lock_table();
    std::unique_ptr<int> key(new int(20));
    std::unique_ptr<int> value(new int(30));
    auto result = lt.insert(std::move(key), std::move(value));
    REQUIRE(*result.first->first == 20);
    REQUIRE(*result.first->second == 20);
    REQUIRE(!result.second);
    REQUIRE(static_cast<bool>(key));
    REQUIRE(static_cast<bool>(value));
  }
}

TEST_CASE("locked_table erase", "[locked_table]") {
  IntIntTable tbl;
  for (int i = 0; i < 5; ++i) {
    tbl.insert(i, i);
  }
  using lt_t = IntIntTable::locked_table;

  SECTION("simple erase") {
    auto lt = tbl.lock_table();
    lt_t::const_iterator const_it;
    const_it = lt.find(0);
    REQUIRE(const_it != lt.end());
    lt_t::const_iterator const_next = const_it;
    ++const_next;
    REQUIRE(static_cast<lt_t::const_iterator>(lt.erase(const_it)) ==
            const_next);
    REQUIRE(lt.size() == 4);

    lt_t::iterator it;
    it = lt.find(1);
    lt_t::iterator next = it;
    ++next;
    REQUIRE(lt.erase(static_cast<lt_t::const_iterator>(it)) == next);
    REQUIRE(lt.size() == 3);

    REQUIRE(lt.erase(2) == 1);
    REQUIRE(lt.size() == 2);
  }

  SECTION("erase doesn't ruin this iterator") {
    auto lt = tbl.lock_table();
    auto it = lt.begin();
    auto next = it;
    ++next;
    REQUIRE(lt.erase(it) == next);
    ++it;
    REQUIRE(it->first > 0);
    REQUIRE(it->first < 5);
    REQUIRE(it->second > 0);
    REQUIRE(it->second < 5);
  }

  SECTION("erase doesn't ruin other iterators") {
    auto lt = tbl.lock_table();
    auto it0 = lt.find(0);
    auto it1 = lt.find(1);
    auto it2 = lt.find(2);
    auto it3 = lt.find(3);
    auto it4 = lt.find(4);
    auto next = it2;
    ++next;
    REQUIRE(lt.erase(it2) == next);
    REQUIRE(it0->first == 0);
    REQUIRE(it0->second == 0);
    REQUIRE(it1->first == 1);
    REQUIRE(it1->second == 1);
    REQUIRE(it3->first == 3);
    REQUIRE(it3->second == 3);
    REQUIRE(it4->first == 4);
    REQUIRE(it4->second == 4);
  }
}

TEST_CASE("locked_table find", "[locked_table]") {
  IntIntTable tbl;
  using lt_t = IntIntTable::locked_table;
  auto lt = tbl.lock_table();
  for (int i = 0; i < 10; ++i) {
    REQUIRE(lt.insert(i, i).second);
  }
  bool found_begin_elem = false;
  bool found_last_elem = false;
  for (int i = 0; i < 10; ++i) {
    lt_t::iterator it = lt.find(i);
    lt_t::const_iterator const_it = lt.find(i);
    REQUIRE(it != lt.end());
    REQUIRE(it->first == i);
    REQUIRE(it->second == i);
    REQUIRE(const_it != lt.end());
    REQUIRE(const_it->first == i);
    REQUIRE(const_it->second == i);
    it->second++;
    if (it == lt.begin()) {
      found_begin_elem = true;
    }
    if (++it == lt.end()) {
      found_last_elem = true;
    }
  }
  REQUIRE(found_begin_elem);
  REQUIRE(found_last_elem);
  for (int i = 0; i < 10; ++i) {
    lt_t::iterator it = lt.find(i);
    REQUIRE(it->first == i);
    REQUIRE(it->second == i + 1);
  }
}

TEST_CASE("locked_table at", "[locked_table]") {
  IntIntTable tbl;
  auto lt = tbl.lock_table();
  for (int i = 0; i < 10; ++i) {
    REQUIRE(lt.insert(i, i).second);
  }
  for (int i = 0; i < 10; ++i) {
    int &val = lt.at(i);
    const int &const_val =
        const_cast<const IntIntTable::locked_table &>(lt).at(i);
    REQUIRE(val == i);
    REQUIRE(const_val == i);
    ++val;
  }
  for (int i = 0; i < 10; ++i) {
    REQUIRE(lt.at(i) == i + 1);
  }
  REQUIRE_THROWS_AS(lt.at(11), std::out_of_range);
}

TEST_CASE("locked_table operator[]", "[locked_table]") {
  IntIntTable tbl;
  auto lt = tbl.lock_table();
  for (int i = 0; i < 10; ++i) {
    REQUIRE(lt.insert(i, i).second);
  }
  for (int i = 0; i < 10; ++i) {
    int &val = lt[i];
    REQUIRE(val == i);
    ++val;
  }
  for (int i = 0; i < 10; ++i) {
    REQUIRE(lt[i] == i + 1);
  }
  REQUIRE(lt[11] == 0);
  REQUIRE(lt.at(11) == 0);
}

TEST_CASE("locked_table count", "[locked_table]") {
  IntIntTable tbl;
  auto lt = tbl.lock_table();
  for (int i = 0; i < 10; ++i) {
    REQUIRE(lt.insert(i, i).second);
  }
  for (int i = 0; i < 10; ++i) {
    REQUIRE(lt.count(i) == 1);
  }
  REQUIRE(lt.count(11) == 0);
}

TEST_CASE("locked_table equal_range", "[locked_table]") {
  IntIntTable tbl;
  using lt_t = IntIntTable::locked_table;
  auto lt = tbl.lock_table();
  for (int i = 0; i < 10; ++i) {
    REQUIRE(lt.insert(i, i).second);
  }
  for (int i = 0; i < 10; ++i) {
    std::pair<lt_t::iterator, lt_t::iterator> it_range = lt.equal_range(i);
    REQUIRE(it_range.first->first == i);
    REQUIRE(++it_range.first == it_range.second);
    std::pair<lt_t::const_iterator, lt_t::const_iterator> const_it_range =
        lt.equal_range(i);
    REQUIRE(const_it_range.first->first == i);
    REQUIRE(++const_it_range.first == const_it_range.second);
  }
  auto it_range = lt.equal_range(11);
  REQUIRE(it_range.first == lt.end());
  REQUIRE(it_range.second == lt.end());
}

TEST_CASE("locked_table rehash", "[locked_table]") {
  IntIntTable tbl(10);
  auto lt = tbl.lock_table();
  REQUIRE(lt.hashpower() == 2);
  lt.rehash(1);
  REQUIRE(lt.hashpower() == 1);
  lt.rehash(10);
  REQUIRE(lt.hashpower() == 10);
}

TEST_CASE("locked_table reserve", "[locked_table]") {
  IntIntTable tbl(10);
  auto lt = tbl.lock_table();
  REQUIRE(lt.hashpower() == 2);
  lt.reserve(1);
  REQUIRE(lt.hashpower() == 0);
  lt.reserve(4096);
  REQUIRE(lt.hashpower() == 10);
}

TEST_CASE("locked_table equality", "[locked_table]") {
  IntIntTable tbl1(40);
  auto lt1 = tbl1.lock_table();
  for (int i = 0; i < 10; ++i) {
    lt1.insert(i, i);
  }

  IntIntTable tbl2(30);
  auto lt2 = tbl2.lock_table();
  for (int i = 0; i < 10; ++i) {
    lt2.insert(i, i);
  }

  IntIntTable tbl3(30);
  auto lt3 = tbl3.lock_table();
  for (int i = 0; i < 10; ++i) {
    lt3.insert(i, i + 1);
  }

  IntIntTable tbl4(40);
  auto lt4 = tbl4.lock_table();
  for (int i = 0; i < 10; ++i) {
    lt4.insert(i + 1, i);
  }

  REQUIRE(lt1 == lt2);
  REQUIRE_FALSE(lt2 != lt1);

  REQUIRE(lt1 != lt3);
  REQUIRE_FALSE(lt3 == lt1);
  REQUIRE_FALSE(lt2 == lt3);
  REQUIRE(lt3 != lt2);

  REQUIRE(lt1 != lt4);
  REQUIRE(lt4 != lt1);
  REQUIRE_FALSE(lt3 == lt4);
  REQUIRE_FALSE(lt4 == lt3);
}

template <typename Table> void check_all_locks_taken(Table &tbl) {
  auto &locks = UnitTestInternalAccess::get_current_locks(tbl);
  for (auto &lock : locks) {
    REQUIRE_FALSE(lock.try_lock());
  }
}

TEST_CASE("locked table holds locks after resize", "[locked table]") {
  IntIntTable tbl(4);
  auto lt = tbl.lock_table();
  check_all_locks_taken(tbl);

  // After a cuckoo_fast_double, all locks are still taken
  for (int i = 0; i < 5; ++i) {
    lt.insert(i, i);
  }
  check_all_locks_taken(tbl);

  // After a cuckoo_simple_expand, all locks are still taken
  lt.rehash(10);
  check_all_locks_taken(tbl);
}

TEST_CASE("locked table IO", "[locked_table]") {
  IntIntTable tbl(0);
  auto lt = tbl.lock_table();
  for (int i = 0; i < 100; ++i) {
    lt.insert(i, i);
  }

  std::stringstream sstream;
  sstream << lt;

  IntIntTable tbl2;
  auto lt2 = tbl2.lock_table();
  sstream.seekg(0);
  sstream >> lt2;

  REQUIRE(100 == lt.size());
  for (int i = 0; i < 100; ++i) {
    REQUIRE(i == lt.at(i));
  }

  REQUIRE(100 == lt2.size());
  for (int i = 100; i < 1000; ++i) {
    lt2.insert(i, i);
  }
  for (int i = 0; i < 1000; ++i) {
    REQUIRE(i == lt2.at(i));
  }
}

TEST_CASE("empty locked table IO", "[locked table]") {
  IntIntTable tbl(0);
  auto lt = tbl.lock_table();
  lt.minimum_load_factor(0.5);
  lt.maximum_hashpower(10);

  std::stringstream sstream;
  sstream << lt;

  IntIntTable tbl2(0);
  auto lt2 = tbl2.lock_table();
  sstream.seekg(0);
  sstream >> lt2;

  REQUIRE(0 == lt.size());
  REQUIRE(0 == lt2.size());
  REQUIRE(0.5 == lt.minimum_load_factor());
  REQUIRE(10 == lt.maximum_hashpower());
}

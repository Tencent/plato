#include <catch.hpp>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

#include "unit_test_util.hh"
#include <libcuckoo/cuckoohash_map.hh>

TEST_CASE("iterator types", "[iterator]") {
  using Ltbl = IntIntTable::locked_table;
  using It = Ltbl::iterator;
  using ConstIt = Ltbl::const_iterator;

  const bool it_difference_type =
      std::is_same<Ltbl::difference_type, It::difference_type>::value;
  const bool it_value_type =
      std::is_same<Ltbl::value_type, It::value_type>::value;
  const bool it_pointer = std::is_same<Ltbl::pointer, It::pointer>::value;
  const bool it_reference = std::is_same<Ltbl::reference, It::reference>::value;
  const bool it_iterator_category =
      std::is_same<std::bidirectional_iterator_tag,
                   It::iterator_category>::value;

  const bool const_it_difference_type =
      std::is_same<Ltbl::difference_type, ConstIt::difference_type>::value;
  const bool const_it_value_type =
      std::is_same<Ltbl::value_type, ConstIt::value_type>::value;
  const bool const_it_reference =
      std::is_same<Ltbl::const_reference, ConstIt::reference>::value;
  const bool const_it_pointer =
      std::is_same<Ltbl::const_pointer, ConstIt::pointer>::value;
  const bool const_it_iterator_category =
      std::is_same<std::bidirectional_iterator_tag,
                   ConstIt::iterator_category>::value;

  REQUIRE(it_difference_type);
  REQUIRE(it_value_type);
  REQUIRE(it_pointer);
  REQUIRE(it_reference);
  REQUIRE(it_iterator_category);

  REQUIRE(const_it_difference_type);
  REQUIRE(const_it_value_type);
  REQUIRE(const_it_pointer);
  REQUIRE(const_it_reference);
  REQUIRE(const_it_iterator_category);
}

TEST_CASE("empty table iteration", "[iterator]") {
  IntIntTable table;
  {
    auto lt = table.lock_table();
    REQUIRE(lt.begin() == lt.begin());
    REQUIRE(lt.begin() == lt.end());

    REQUIRE(lt.cbegin() == lt.begin());
    REQUIRE(lt.begin() == lt.end());

    REQUIRE(lt.cbegin() == lt.begin());
    REQUIRE(lt.cend() == lt.end());
  }
}

TEST_CASE("iterator walkthrough", "[iterator]") {
  IntIntTable table;
  for (int i = 0; i < 10; ++i) {
    table.insert(i, i);
  }

  SECTION("forward postfix walkthrough") {
    auto lt = table.lock_table();
    auto it = lt.cbegin();
    for (size_t i = 0; i < table.size(); ++i) {
      REQUIRE((*it).first == (*it).second);
      REQUIRE(it->first == it->second);
      auto old_it = it;
      REQUIRE(old_it == it++);
    }
    REQUIRE(it == lt.end());
  }

  SECTION("forward prefix walkthrough") {
    auto lt = table.lock_table();
    auto it = lt.cbegin();
    for (size_t i = 0; i < table.size(); ++i) {
      REQUIRE((*it).first == (*it).second);
      REQUIRE(it->first == it->second);
      ++it;
    }
    REQUIRE(it == lt.end());
  }

  SECTION("backwards postfix walkthrough") {
    auto lt = table.lock_table();
    auto it = lt.cend();
    for (size_t i = 0; i < table.size(); ++i) {
      auto old_it = it;
      REQUIRE(old_it == it--);
      REQUIRE((*it).first == (*it).second);
      REQUIRE(it->first == it->second);
    }
    REQUIRE(it == lt.begin());
  }

  SECTION("backwards prefix walkthrough") {
    auto lt = table.lock_table();
    auto it = lt.cend();
    for (size_t i = 0; i < table.size(); ++i) {
      --it;
      REQUIRE((*it).first == (*it).second);
      REQUIRE(it->first == it->second);
    }
    REQUIRE(it == lt.begin());
  }

  SECTION("walkthrough works after move") {
    auto lt = table.lock_table();
    auto it = lt.cend();
    auto lt2 = std::move(lt);
    for (size_t i = 0; i < table.size(); ++i) {
      --it;
      REQUIRE((*it).first == (*it).second);
      REQUIRE(it->first == it->second);
    }
    REQUIRE(it == lt2.begin());
  }
}

TEST_CASE("iterator modification", "[iterator]") {
  IntIntTable table;
  for (int i = 0; i < 10; ++i) {
    table.insert(i, i);
  }

  auto lt = table.lock_table();
  for (auto it = lt.begin(); it != lt.end(); ++it) {
    it->second = it->second + 1;
  }

  auto it = lt.cbegin();
  for (size_t i = 0; i < table.size(); ++i) {
    REQUIRE(it->first == it->second - 1);
    ++it;
  }
  REQUIRE(it == lt.end());
}

TEST_CASE("lock table blocks inserts", "[iterator]") {
  IntIntTable table;
  auto lt = table.lock_table();
  std::thread thread([&table]() {
    for (int i = 0; i < 10; ++i) {
      table.insert(i, i);
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  REQUIRE(table.size() == 0);
  lt.unlock();
  thread.join();

  REQUIRE(table.size() == 10);
}

TEST_CASE("Cast iterator to const iterator", "[iterator]") {
  IntIntTable table;
  for (int i = 0; i < 10; ++i) {
    table.insert(i, i);
  }
  auto lt = table.lock_table();
  for (IntIntTable::locked_table::iterator it = lt.begin(); it != lt.end();
       ++it) {
    REQUIRE(it->first == it->second);
    it->second++;
    IntIntTable::locked_table::const_iterator const_it = it;
    REQUIRE(it->first + 1 == it->second);
  }
}

#include <catch.hpp>
#include <cerrno>
#include <cstdio>

extern "C" {
#include "int_int_table.h"
}

int cuckoo_find_fn_value;
void cuckoo_find_fn(const int *value) { cuckoo_find_fn_value = *value; }

void cuckoo_increment_fn(int *value) { ++(*value); }

bool cuckoo_erase_fn(int *value) { return (*value) & 1; }

TEST_CASE("c interface", "[c interface]") {
  int_int_table *tbl = int_int_table_init(0);

  SECTION("empty table statistics") {
    REQUIRE(int_int_table_hashpower(tbl) == 0);
    REQUIRE(int_int_table_bucket_count(tbl) == 1);
    REQUIRE(int_int_table_empty(tbl));
    REQUIRE(int_int_table_capacity(tbl) == 4);
    REQUIRE(int_int_table_load_factor(tbl) == 0);
  }

  for (int i = 0; i < 10; i++) {
    int_int_table_insert(tbl, &i, &i);
  }

  SECTION("find_fn") {
    for (int i = 0; i < 10; ++i) {
      REQUIRE(int_int_table_find_fn(tbl, &i, cuckoo_find_fn));
      REQUIRE(cuckoo_find_fn_value == i);
    }
    for (int i = 10; i < 20; ++i) {
      REQUIRE_FALSE(int_int_table_find_fn(tbl, &i, cuckoo_find_fn));
    }
  }

  SECTION("update_fn") {
    for (int i = 0; i < 10; ++i) {
      REQUIRE(int_int_table_update_fn(tbl, &i, cuckoo_increment_fn));
    }
    for (int i = 0; i < 10; ++i) {
      REQUIRE(int_int_table_find_fn(tbl, &i, cuckoo_find_fn));
      REQUIRE(cuckoo_find_fn_value == i + 1);
    }
  }

  SECTION("upsert") {
    for (int i = 0; i < 10; ++i) {
      REQUIRE_FALSE(int_int_table_upsert(tbl, &i, cuckoo_increment_fn, &i));
      REQUIRE(int_int_table_find_fn(tbl, &i, cuckoo_find_fn));
      REQUIRE(cuckoo_find_fn_value == i + 1);
    }
    for (int i = 10; i < 20; ++i) {
      REQUIRE(int_int_table_upsert(tbl, &i, cuckoo_increment_fn, &i));
      REQUIRE(int_int_table_find_fn(tbl, &i, cuckoo_find_fn));
      REQUIRE(cuckoo_find_fn_value == i);
    }
  }

  SECTION("erase_fn") {
    for (int i = 0; i < 10; ++i) {
      if (i & 1) {
        REQUIRE(int_int_table_erase_fn(tbl, &i, cuckoo_erase_fn));
        REQUIRE_FALSE(int_int_table_find_fn(tbl, &i, cuckoo_find_fn));
      } else {
        REQUIRE(int_int_table_erase_fn(tbl, &i, cuckoo_erase_fn));
        REQUIRE(int_int_table_find_fn(tbl, &i, cuckoo_find_fn));
      }
    }
  }

  SECTION("find") {
    int value;
    for (int i = 0; i < 10; ++i) {
      REQUIRE(int_int_table_find(tbl, &i, &value));
      REQUIRE(value == i);
    }
    for (int i = 10; i < 20; ++i) {
      REQUIRE_FALSE(int_int_table_find(tbl, &i, &value));
    }
  }

  SECTION("contains") {
    for (int i = 0; i < 10; ++i) {
      REQUIRE(int_int_table_contains(tbl, &i));
    }
    for (int i = 10; i < 20; ++i) {
      REQUIRE_FALSE(int_int_table_contains(tbl, &i));
    }
  }

  SECTION("update") {
    int value;
    for (int i = 0; i < 10; ++i) {
      int new_value = i + 1;
      REQUIRE(int_int_table_update(tbl, &i, &new_value));
      REQUIRE(int_int_table_find(tbl, &i, &value));
      REQUIRE(value == i + 1);
    }
    for (int i = 10; i < 20; ++i) {
      REQUIRE_FALSE(int_int_table_update(tbl, &i, &value));
    }
  }

  SECTION("insert_or_assign") {
    for (int i = 0; i < 10; ++i) {
      REQUIRE_FALSE(int_int_table_insert_or_assign(tbl, &i, &i));
    }
    for (int i = 10; i < 20; ++i) {
      REQUIRE(int_int_table_insert_or_assign(tbl, &i, &i));
    }
    for (int i = 0; i < 20; ++i) {
      int value;
      REQUIRE(int_int_table_find(tbl, &i, &value));
      REQUIRE(value == i);
    }
  }

  SECTION("erase") {
    for (int i = 1; i < 10; i += 2) {
      REQUIRE(int_int_table_erase(tbl, &i));
    }
    for (int i = 0; i < 10; ++i) {
      REQUIRE(int_int_table_contains(tbl, &i) != (i & 1));
    }
  }

  SECTION("rehash") {
    REQUIRE(int_int_table_rehash(tbl, 15));
    REQUIRE(int_int_table_hashpower(tbl) == 15);
  }

  SECTION("reserve") {
    REQUIRE(int_int_table_reserve(tbl, 30));
    REQUIRE(int_int_table_hashpower(tbl) == 3);
  }

  SECTION("clear") {
    int_int_table_clear(tbl);
    REQUIRE(int_int_table_empty(tbl));
  }

  SECTION("read/write") {
    FILE *fp = tmpfile();
    int_int_table_locked_table *ltbl = int_int_table_lock_table(tbl);
    REQUIRE(int_int_table_locked_table_write(ltbl, fp));
    rewind(fp);
    int_int_table *tbl2 = int_int_table_read(fp);
    REQUIRE(int_int_table_size(tbl2) == 10);
    for (int i = 0; i < 10; ++i) {
      int value;
      REQUIRE(int_int_table_find(tbl2, &i, &value));
      REQUIRE(i == value);
    }
    int_int_table_free(tbl2);
    int_int_table_locked_table_free(ltbl);
    fclose(fp);
  }

  int_int_table_free(tbl);
}

TEST_CASE("c interface locked table", "[c interface]") {
  int_int_table *tbl = int_int_table_init(0);
  int_int_table_locked_table *ltbl = int_int_table_lock_table(tbl);

  SECTION("is_active/unlock") {
    REQUIRE(int_int_table_locked_table_is_active(ltbl));
    int_int_table_locked_table_unlock(ltbl);
    REQUIRE_FALSE(int_int_table_locked_table_is_active(ltbl));
  }

  SECTION("statistics") {
    REQUIRE(int_int_table_locked_table_hashpower(ltbl) == 0);
    REQUIRE(int_int_table_locked_table_bucket_count(ltbl) == 1);
    REQUIRE(int_int_table_locked_table_empty(ltbl));
    REQUIRE(int_int_table_locked_table_size(ltbl) == 0);
    REQUIRE(int_int_table_locked_table_capacity(ltbl) == 4);
    REQUIRE(int_int_table_locked_table_load_factor(ltbl) == 0);
  }

  for (int i = 0; i < 10; ++i) {
    int_int_table_locked_table_insert(ltbl, &i, &i, NULL);
  }

  SECTION("constant iteration") {
    int occurrences[10] = {};
    int_int_table_const_iterator *begin =
        int_int_table_locked_table_cbegin(ltbl);
    int_int_table_const_iterator *end = int_int_table_locked_table_cend(ltbl);
    for (; !int_int_table_const_iterator_equal(begin, end);
         int_int_table_const_iterator_increment(begin)) {
      ++occurrences[*int_int_table_const_iterator_key(begin)];
      REQUIRE(*int_int_table_const_iterator_key(begin) ==
              *int_int_table_const_iterator_mapped(begin));
    }
    for (int i = 0; i < 10; ++i) {
      REQUIRE(occurrences[i] == 1);
    }
    int_int_table_const_iterator_decrement(end);
    int_int_table_const_iterator_set(begin, end);
    REQUIRE(int_int_table_const_iterator_equal(begin, end));
    int_int_table_locked_table_set_cbegin(ltbl, begin);
    for (; !int_int_table_const_iterator_equal(end, begin);
         int_int_table_const_iterator_decrement(end)) {
      ++occurrences[*int_int_table_const_iterator_key(end)];
    }
    ++occurrences[*int_int_table_const_iterator_key(end)];
    for (int i = 0; i < 10; ++i) {
      REQUIRE(occurrences[i] == 2);
    }
    int_int_table_const_iterator_free(end);
    int_int_table_const_iterator_free(begin);
  }

  SECTION("iteration") {
    int_int_table_iterator *begin = int_int_table_locked_table_begin(ltbl);
    int_int_table_iterator *end = int_int_table_locked_table_end(ltbl);
    for (; !int_int_table_iterator_equal(begin, end);
         int_int_table_iterator_increment(begin)) {
      ++(*int_int_table_iterator_mapped(begin));
    }
    int_int_table_iterator_set(begin, end);
    REQUIRE(int_int_table_iterator_equal(begin, end));
    int_int_table_locked_table_set_begin(ltbl, begin);
    for (; !int_int_table_iterator_equal(begin, end);
         int_int_table_iterator_increment(begin)) {
      REQUIRE(*int_int_table_iterator_key(begin) + 1 ==
              *int_int_table_iterator_mapped(begin));
    }
    int_int_table_iterator_free(end);
    int_int_table_iterator_free(begin);
  }

  SECTION("clear") {
    int_int_table_locked_table_clear(ltbl);
    REQUIRE(int_int_table_locked_table_size(ltbl) == 0);
  }

  SECTION("insert with iterator") {
    int_int_table_iterator *it = int_int_table_locked_table_begin(ltbl);
    int item = 11;
    REQUIRE(int_int_table_locked_table_insert(ltbl, &item, &item, it));
    REQUIRE(*int_int_table_iterator_key(it) == 11);
    REQUIRE(*int_int_table_iterator_mapped(it) == 11);
    item = 5;
    REQUIRE_FALSE(int_int_table_locked_table_insert(ltbl, &item, &item, it));
    REQUIRE(*int_int_table_iterator_key(it) == 5);
    ++(*int_int_table_iterator_mapped(it));
    REQUIRE(*int_int_table_iterator_mapped(it) == 6);
    int_int_table_iterator_free(it);
  }

  SECTION("erase") {
    int_int_table_iterator *it1 = int_int_table_locked_table_begin(ltbl);
    int_int_table_iterator *it2 = int_int_table_locked_table_begin(ltbl);
    int_int_table_iterator_increment(it2);

    int_int_table_locked_table_erase_it(ltbl, it1, it1);
    REQUIRE(int_int_table_iterator_equal(it1, it2));

    int_int_table_const_iterator *cbegin =
        int_int_table_locked_table_cbegin(ltbl);
    int_int_table_iterator_increment(it2);

    int_int_table_locked_table_erase_const_it(ltbl, cbegin, it1);
    REQUIRE(int_int_table_iterator_equal(it1, it2));

    int_int_table_const_iterator_free(cbegin);
    int_int_table_iterator_free(it2);
    int_int_table_iterator_free(it1);

    int successes = 0;
    for (int i = 0; i < 10; ++i) {
      successes += int_int_table_locked_table_erase(ltbl, &i);
    }
    REQUIRE(successes == 8);
    REQUIRE(int_int_table_locked_table_empty(ltbl));
  }

  SECTION("find") {
    int_int_table_iterator *it = int_int_table_locked_table_begin(ltbl);
    int_int_table_const_iterator *cit = int_int_table_locked_table_cbegin(ltbl);

    int item = 0;
    int_int_table_locked_table_find(ltbl, &item, it);
    REQUIRE(*int_int_table_iterator_key(it) == 0);
    REQUIRE(*int_int_table_iterator_mapped(it) == 0);
    item = 10;
    int_int_table_locked_table_find_const(ltbl, &item, cit);
    int_int_table_const_iterator *cend = int_int_table_locked_table_cend(ltbl);
    REQUIRE(int_int_table_const_iterator_equal(cit, cend));

    int_int_table_const_iterator_free(cend);
    int_int_table_const_iterator_free(cit);
    int_int_table_iterator_free(it);
  }

  SECTION("rehash") {
    int_int_table_locked_table_rehash(ltbl, 15);
    REQUIRE(int_int_table_locked_table_hashpower(ltbl) == 15);
  }

  SECTION("reserve") {
    int_int_table_locked_table_reserve(ltbl, 30);
    REQUIRE(int_int_table_locked_table_hashpower(ltbl) == 3);
  }

  int_int_table_locked_table_free(ltbl);
  int_int_table_free(tbl);
}

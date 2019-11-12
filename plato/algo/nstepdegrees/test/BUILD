load("//build_tools/rules:variables.bzl", "PLATO_OPTS")

cc_test (
  name = "test_nstepdegrees",
  srcs = [
    "test_nstepdegrees.cc",
  ],
  copts = ['-O2', ] + PLATO_OPTS,
  linkopts = [ ] + PLATO_OPTS,
  deps = [
    "//3rd/googletest:gtest_main",
    "//3rd/gflags:gflags",
    "//plato/util:gtest_mpi",

    "//3rd/jemalloc:jemalloc",
    "//plato/graph:graph",
    "//plato/util:perf",
    "//plato/util:atomic",
    "//plato/engine:dualmode",
    "//plato/algo/nstepdegrees:nstepdegrees",
  ],
  data = [
    "//data:graph/raw_graph_7_7.csv",
    "//data:graph/raw_graph_2.csv",
  ],
  linkstatic = 1,
  timeout = "short",
)

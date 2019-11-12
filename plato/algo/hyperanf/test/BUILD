load("//build_tools/rules:variables.bzl", "PLATO_OPTS")

cc_test (
  name = "test_hyperanf",
  srcs = [
    "test_hyperanf.cc",
  ],
  copts = ['-O2', ] + PLATO_OPTS,
  linkopts = [ ] + PLATO_OPTS,
  deps = [
    "//3rd/googletest:gtest_main",
    "//3rd/gflags:gflags",
    "//plato/util:gtest_mpi",
    "//plato/algo/hyperanf:hyperanf",
    "//plato/util:spinlock",
  ],
  data = [
    "//data:graph/raw_graph_7_7.csv",
  ],
  linkstatic = 1,
  timeout = "short",
)


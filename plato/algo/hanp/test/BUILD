load("//build_tools/rules:variables.bzl", "PLATO_OPTS")

cc_test (
  name = "test_hanp",
  srcs = [
    "test_hanp.cc",
  ],
  copts = ['-O2', ] + PLATO_OPTS,
  linkopts = [ ] + PLATO_OPTS,
  deps = [
    "//3rd/googletest:gtest_main",
    "//3rd/gflags:gflags",
    "//plato/util:gtest_mpi",
    "//plato/algo/hanp:hanp",
  ],
  data = [
    "//data:graph/raw_graph_10_9_weighted.csv",
  ],
  linkstatic = 1,
  timeout = "short",
)

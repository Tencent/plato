load("//build_tools/rules:variables.bzl", "PLATO_OPTS")

cc_test (
  name = "test_bnc",
  srcs = [
    "test_bnc.cc",
  ],
  copts = ['-O2', ] + PLATO_OPTS,
  linkopts = [ ] + PLATO_OPTS,
  deps = [
    "//3rd/googletest:gtest_main",
    "//3rd/gflags:gflags",
    "//plato/util:gtest_mpi",
    "//plato/algo/bnc:bnc",
  ],
  data = [
    "//data:graph/v100_e2150_ua_c3.csv",
  ],
  linkstatic = 1,
  timeout = "short",
)

load("//build_tools/rules:variables.bzl", "PLATO_OPTS")

cc_test (
  name = "test_cgm",
  srcs = [
    "test_cgm.cc",
  ],
  copts = ['-O2', ] + PLATO_OPTS,
  linkopts = [ ] + PLATO_OPTS,
  deps = [
    "//3rd/googletest:gtest_main",
    "//3rd/gflags:gflags",
    "//plato/util:gtest_mpi",
    "//plato/algo/cgm:cgm",
  ],
  data = [
    "//data:graph/v100_e2150_ua_c3.csv",
  ],
  linkstatic = 1,
  timeout = "short",
)

load("//build_tools/rules:variables.bzl", "PLATO_OPTS")

cc_test (
    name = "edge_cache",
    srcs = [
        "edge_cache.cc",
    ],
    copts = [ '-Wall', '-std=c++11', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/googletest:gtest_main",

        "//plato/graph/structure:edge_cache",
    ],
    data = [ ],
    linkstatic = 1,
    timeout = "short",
)

cc_test (
    name = "bcsr",
    srcs = [
        "bcsr.cc",
    ],
    copts = [ '-Wall', '-Werror', '-std=c++11', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/googletest:gtest",

        "//plato/util:gtest_mpi",
        "//plato/util:bitmap",
        "//plato/graph:base",
        "//plato/graph:structure",
        "//plato/graph/structure:bcsr",
        "//plato/graph/structure:edge_cache",
        "//plato/graph/partition:hash",
        "//plato/graph/partition:sequence",
    ],
    data = [
      "//data:graph/v100_e4300_da_c3.csv",
      "//data:graph/v100_e2150_ua_c3.csv",
      "//data:graph/graph_10_9.csv",
    ],
    defines = [
        "__EDGE_CACHE_DEBUG__",
    ],
    linkstatic = 1,
    timeout = "moderate",
)

cc_test (
    name = "cbcsr",
    srcs = [
        "cbcsr.cc",
    ],
    copts = [ '-Wall', '-Werror', '-std=c++11', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/googletest:gtest",

        "//plato/util:gtest_mpi",
        "//plato/util:bitmap",
        "//plato/graph:base",
        "//plato/graph:structure",
        "//plato/graph/structure:cbcsr",
        "//plato/graph/structure:edge_cache",
        "//plato/graph/partition:hash",
        "//plato/graph/partition:sequence",
    ],
    data = [
      "//data:graph/v100_e4300_da_c3.csv",
      "//data:graph/v100_e2150_ua_c3.csv",
      "//data:graph/graph_10_9.csv",
      "//data:graph/non_coding_5_8.csv",
      "//data:graph/non_coding_5_8_weighted.csv",
      "//data:graph/non_coding_5_7.csv",
      "//data:graph/non_coding_5_7_weighted.csv",
    ],
    defines = [
        "__EDGE_CACHE_DEBUG__",
    ],
    linkstatic = 1,
    timeout = "moderate",
)

cc_test (
    name = "hnbbcsr",
    srcs = [
        "hnbbcsr.cc",
    ],
    copts = [ '-Wall', '-Werror', '-std=c++11', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/googletest:gtest",

        "//plato/util:gtest_mpi",
        "//plato/util:bitmap",
        "//plato/graph:base",
        "//plato/graph:structure",
        "//plato/graph/structure:hnbbcsr",
        "//plato/graph/structure:edge_cache",
        "//plato/graph/partition:hash",
        "//plato/graph/partition:sequence",
    ],
    data = [
      "//data:graph/v100_e4300_da_c3.csv",
      "//data:graph/v100_e2150_ua_c3.csv",
      "//data:graph/graph_10_9.csv",
      "//data:graph/non_coding_5_7.csv",
      "//data:graph/non_coding_5_7_weighted.csv",
      "//data:graph/non_coding_5_8.csv",
      "//data:graph/non_coding_5_8_weighted.csv",
      "//data:graph/non_coding_5_8_vertic.csv",
    ],
    defines = [
        "__EDGE_CACHE_DEBUG__",
    ],
    linkstatic = 1,
    timeout = "moderate",
)

cc_test (
    name = "dcsc",
    srcs = [
        "dcsc.cc",
    ],
    copts = [ '-Wall', '-Werror', '-std=c++11', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/googletest:gtest",

        "//plato/util:gtest_mpi",
        "//plato/util:bitmap",
        "//plato/graph:base",
        "//plato/graph:structure",
        "//plato/graph/structure:bcsr",
        "//plato/graph/structure:dcsc",
        "//plato/graph/structure:edge_cache",
        "//plato/graph/partition:hash",
        "//plato/graph/partition:sequence",
    ],
    data = [
      "//data:graph/v100_e4300_da_c3.csv",
      "//data:graph/v100_e2150_ua_c3.csv",
      "//data:graph/graph_10_9.csv",
    ],
    linkstatic = 1,
    timeout = "moderate",
)

cc_test (
    name = "tcsr",
    srcs = [
        "tcsr.cc",
    ],
    copts = [ '-Wall', '-Werror', '-std=c++11', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/googletest:gtest",

        "//plato/util:gtest_mpi",
        "//plato/util:bitmap",
        "//plato/graph:base",
        "//plato/graph:structure",
        "//plato/graph/structure:tcsr",
        "//plato/graph/partition:hash",
        "//plato/graph/partition:sequence",
    ],
    data = [
      "//data:graph/v100_e4300_da_c3.csv",
      "//data:graph/v100_e2150_ua_c3.csv",
      "//data:graph/graph_10_9.csv",
    ],
    defines = [
        "__EDGE_CACHE_DEBUG__",
    ],
    linkstatic = 1,
    timeout = "moderate",
)

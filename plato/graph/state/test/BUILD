load("//build_tools/rules:variables.bzl", "PLATO_OPTS")

cc_test (
    name = "dense_state",
    srcs = [
        "dense_state.cc",
    ],
    copts = [ '-Wall', '-Werror', '-std=c++11', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/googletest:gtest",

        "//plato/util:gtest_mpi",
        "//plato/util:bitmap",
        "//plato/util:mmap_alloc",
        "//plato/graph:structure",
        "//plato/graph/partition:hash",
        "//plato/graph/partition:sequence",
        "//plato/graph/state:dense_state",
    ],
    data = [
      "//data:graph/v100_e4300_da_c3.csv",
      "//data:graph/v100_e2150_ua_c3.csv",
      "//data:graph/graph_10_9.csv",
    ],
    linkstatic = 1,
    timeout = "short",
)

cc_test (
    name = "sparse_state",
    srcs = [
        "sparse_state.cc",
    ],
    copts = [ '-Wall', '-Werror', '-std=c++11', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/googletest:gtest",

        "//plato/util:gtest_mpi",
        "//plato/util:bitmap",
        "//plato/graph:structure",
        "//plato/graph/partition:hash",
        "//plato/graph/partition:sequence",
        "//plato/graph/state:sparse_state",
    ],
    data = [
      "//data:graph/v100_e4300_da_c3.csv",
      "//data:graph/v100_e2150_ua_c3.csv",
      "//data:graph/graph_10_9.csv",
    ],
    linkstatic = 1,
    timeout = "short",
)

cc_test (
    name = "allgather_state",
    srcs = [
        "allgather_state.cc",
    ],
    copts = [ '-Wall', '-Werror', '-std=c++11', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/googletest:gtest",

        "//plato/util:gtest_mpi",
        "//plato/util:bitmap",
        "//plato/util:mmap_alloc",
        "//plato/graph:structure",
        "//plato/graph/partition:hash",
        "//plato/graph/partition:sequence",
        "//plato/graph/state:dense_state",
    ],
    data = [
      "//data:graph/v100_e4300_da_c3.csv",
      "//data:graph/v100_e2150_ua_c3.csv",
      "//data:graph/graph_10_9.csv",
    ],
    linkstatic = 1,
    timeout = "short",
)

cc_test (
    name = "vertex_cache",
    srcs = [
        "vertex_cache.cc",
    ],
    copts = [ '-Wall', '-std=c++11', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/googletest:gtest_main",

        "//plato/graph/state:vertex_cache",
    ],
    data = [ ],
    linkstatic = 1,
    timeout = "short",
)


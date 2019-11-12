load("//build_tools/rules:variables.bzl", "PLATO_OPTS")

cc_test (
    name = "walk",
    srcs = [
        "walk.cc",
    ],
    copts = [ '-O2', '-Wall', "-Werror", '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/gflags:gflags",
        "//3rd/googletest:gtest",
        "//plato/graph/structure:hnbbcsr",
        "//plato/util:gtest_mpi",
        "//plato/util:spinlock",
        "//plato/engine:walk",
    ],
    defines = [ ],
    data = [
        "//data:graph/graph_10_9.csv",
        "//data:graph/v100_e2150_ua_c3.csv",
        "//data:graph/v100_e4300_da_c3.csv",
        "//data:graph/v100_types.csv",
        "//data:graph/non_coding_5_7.csv",
        "//data:graph/non_coding_5_7_weighted.csv",
        "//data:graph/non_coding_5_8.csv",
        "//data:graph/non_coding_5_8_weighted.csv",
        "//data:graph/non_coding_5_8_vertic.csv",
    ],
    linkstatic = 1,
    timeout = "short",
)


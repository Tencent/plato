load("//build_tools/rules:variables.bzl", "PLATO_OPTS")

cc_test (
    name = "parsers",
    srcs = [
        "parsers.cc",
    ],
    copts = [ '-Wall', '-Werror', '-std=c++11', '-O2', ],
    linkopts = [ ],
    deps = [
        "//3rd/googletest:gtest_main",

        "//plato/graph:parsers",
    ],
    data = [
        "//data:graph/raw_graph_10_9.csv",
        "//data:graph/graph_10_9.csv",
    ],
    linkstatic = 1,
    timeout = "short",
)

# cc_test (
#     name = "adjlist",
#     srcs = [
#         "adjlist.cc",
#     ],
#     copts = [ '-O2', '-Wall', "-Werror", '-std=c++11', ],
#     linkopts = [ ],
#     deps = [
#         "//3rd/googletest:gtest",
#         "//3rd/googletest:gtest_main",
# 
#         "//graph/detail:storage",
#     ],
#     data = [
#         "//data:graph/v100_e2150_ua_c3.bin",
#         "//data:graph/v100_e4300_da_c3.bin",
#         "//data:graph/v100_uint32.csv",
#     ],
#     linkstatic = 1,
#     timeout = "short",
# )

cc_test (
    name = "structure",
    srcs = [
        "structure.cc",
    ],
    copts = [ '-O2', '-Wall', "-Werror", '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/gflags:gflags",
        "//3rd/googletest:gtest",

        "//plato/util:gtest_mpi",
        "//plato/graph/structure:edge_cache",
        "//plato/graph:structure",
    ],
    defines = [
        "__PLATO_DEBUG__",
        "__EDGE_CACHE_DEBUG__",
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
    name = "message_passing",
    srcs = [
        "message_passing.cc",
    ],
    copts = [ '-O2', '-Wall', "-Werror", '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/gflags:gflags",
        "//3rd/googletest:gtest",

        "//plato/util:gtest_mpi",
        "//plato/graph:structure",
        "//plato/graph:message_passing",
        "//plato/util:spinlock",
    ],
    defines = [ ],
    data = [
        "//data:graph/graph_10_9.csv",
        "//data:graph/v100_e2150_ua_c3.csv",
    ],
    linkstatic = 1,
    timeout = "short",
)

# cc_test (
#     name = 'graph',
#     srcs = [
#         'graph.cc',
#     ],
#     copts = [ '-O2', '-Wall', "-Werror", '-std=c++11', ] + PLATO_OPTS,
#     linkopts = [ ] + PLATO_OPTS,
#     deps = [
#         "//3rd/gflags:gflags",
#         "//3rd/googletest:gtest",
# 
#         "//util:gtest_mpi",
#         "//graph:graph",
#     ],
#     defines = [
#         "__PLATO_DEBUG__",
#     ],
#     data = [
#         "//data:graph/v100_e4300_da_c3.csv",
#     ],
#     linkstatic = 1,
#     timeout = "short",
# )


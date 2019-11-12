load("//build_tools/rules:variables.bzl", "PLATO_OPTS")

cc_library (
    name = "detail",
    hdrs = [
        "detail.hpp",
    ],
    srcs = [ ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",

        "//plato/graph:base",
        "//plato/parallel:mpi",
        "//plato/util:bitmap",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)

cc_library (
    name = "sparse_state",
    hdrs = [
        "sparse_state.hpp",
    ],
    srcs = [ ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//plato/graph:base",
        "//plato/graph/partition:sequence",
        "//plato/util:hash",
        "//plato/util/libcuckoo:libcuckoo",

        ":detail",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)

cc_library (
    name = "dense_state",
    hdrs = [
        "dense_state.hpp",
        "allgather_state.hpp",
    ],
    srcs = [ ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//plato/graph:base",
        "//plato/graph/partition:sequence",
        "//plato/util:bitmap",
        "//plato/util:mmap_alloc",

        ":detail",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)

cc_library (
    name = "view",
    hdrs = [
        "view.hpp",
    ],
    srcs = [ ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",

        "//plato/graph:base",
        "//plato/util:bitmap",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)

cc_library (
    name = "vertex_cache",
    hdrs = [
        "vertex_cache.hpp",
    ],
    srcs = [ ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//plato/graph:base",
        "//plato/util:object_buffer",
        "//plato/util:mmap_alloc",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)

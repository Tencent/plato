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
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)

cc_library (
    name = "edge_cache",
    hdrs = [
        "edge_cache.hpp",
    ],
    srcs = [ ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//plato/graph:base",
        "//plato/util:object_buffer",
        "//plato/util:mmap_alloc",

        ":detail",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)

cc_library (
    name = "bcsr",
    hdrs = [
        "bcsr.hpp",
    ],
    srcs = [ ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//plato/graph:base",
        "//plato/graph/partition:sequence",
        "//plato/parallel:mpi",
        "//plato/parallel:bsp",
        "//plato/util:perf",
        "//plato/util:mmap_alloc",

        ":detail"
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)

cc_library (
    name = "cbcsr",
    hdrs = [
        "cbcsr.hpp",
    ],
    srcs = [ ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//plato/graph:base",
        "//plato/graph/partition:sequence",
        "//plato/parallel:mpi",
        "//plato/parallel:bsp",
        "//plato/util:perf",
        "//plato/util:mmap_alloc",

        ":detail"
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)

cc_library (
    name = "hnbbcsr",
    hdrs = [
        "hnbbcsr.hpp",
    ],
    srcs = [ ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//plato/graph:base",
        "//plato/graph/partition:sequence",
        "//plato/parallel:mpi",
        "//plato/parallel:bsp",
        "//plato/util:perf",
        "//plato/util:mmap_alloc",

        ":detail"
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)

cc_library (
    name = "dcsc",
    hdrs = [
        "dcsc.hpp",
    ],
    srcs = [ ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//plato/graph:base",
        "//plato/graph/partition:sequence",
        "//plato/parallel:mpi",
        "//plato/parallel:bsp",
        "//plato/util:perf",
        "//plato/util:mmap_alloc",

        ":detail"
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)

cc_library (
    name = "tcsr",
    hdrs = [
        "tcsr.hpp",
    ],
    srcs = [ ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",
        "//3rd/boost:boost_include",

        "//plato/graph:base",
        "//plato/parallel:mpi",
        "//plato/parallel:bsp",
        "//plato/graph/state:sparse_state",
        "//plato/util:defer",
        "//plato/util:bitmap",
        "//plato/util:archive",
        "//plato/util:hash",
        "//plato/graph/partition:hash",

        ":detail",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)

cc_library (
    name = "vid_encoder",
    hdrs = [
        "vid_encoder.hpp",
    ],
    srcs = [ ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//plato/graph:base",
        "//plato/graph:message_passing",
        "//plato/util/libcuckoo:libcuckoo",
        "//plato/util:perf",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)

load("//build_tools/rules:variables.bzl", "PLATO_OPTS")

cc_library (
    name = "base",
    hdrs = [
        "base.hpp",
    ],
    srcs = [
        "base.cc",
    ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//plato/util:backtrace",
        "//plato/util/libcuckoo:libcuckoo",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ '-lnuma', ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)

cc_library (
    name = "detail",
    hdrs = [
        "detail.hpp",
    ],
    srcs = [ ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",
        "//plato/parallel:shuffle",
        "//3rd/boost:boost_include",
        "//3rd/boost:boost_filesystem",

        ":base",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)

cc_library (
    name = "parsers",
    hdrs = [
        "parsers.hpp",
    ],
    srcs = [ ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",
        "//3rd/boost:boost_include",

        "//plato/graph:base",
        "//plato/util:libsvm",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ],
    linkopts = [ ],
    visibility = ["//visibility:public"],
)

cc_library (
    name = "structure",
    hdrs = [
        "structure.hpp",
    ],
    srcs = [ ],
    includes = [ ".", ],
    deps = [
        "//3rd/yas:yas",
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//plato/util:hdfs",
        "//plato/util:perf",
        "//plato/util:bitmap",
        "//plato/util:spinlock",
        "//plato/util:mmap_alloc",
        "//plato/parallel:mpi",
        "//plato/parallel:bsp",
        "//plato/graph:base",
        "//plato/graph:state",
        "//plato/graph/structure:bcsr",
        "//plato/graph/structure:dcsc",
        "//plato/graph/structure:tcsr",
        "//plato/graph/structure:edge_cache",
        "//plato/graph/structure:vid_encoder",
        "//plato/graph/partition:hash",
        "//plato/graph/partition:dummy",
        "//plato/graph/partition:sequence",

        ":detail",
        ":parsers",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)

cc_library (
    name = "message_passing",
    hdrs = [
        "message_passing.hpp",
    ],
    srcs = [ ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//plato/graph:base",
        "//plato/util:bitmap",
        "//plato/util:spinlock",
        "//plato/parallel:mpi",
        "//plato/parallel:bsp",
        "//plato/parallel:broadcast",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)

cc_library (
    name = "state",
    hdrs = [
        "state.hpp",
    ],
    srcs = [ ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//plato/graph:base",
        "//plato/graph/state:view",
        "//plato/graph/state:dense_state",
        "//plato/graph/state:sparse_state",
        "//plato/graph/state:vertex_cache",
        "//plato/util:hdfs",
        "//plato/util:perf",
        "//plato/util:libsvm",
        "//plato/util:bitmap",
        "//plato/util:spinlock",
        "//plato/util:mmap_alloc",
        "//plato/parallel:mpi",
        "//plato/parallel:bsp",

        ":detail",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)

cc_library (
    name = "graph",
    hdrs = [
        "graph.hpp",
    ],
    srcs = [ ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        ":base",
        ":state",
        ":structure",
        ":message_passing",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)


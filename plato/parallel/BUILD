load("//build_tools/rules:variables.bzl", "PLATO_OPTS")

cc_library (
    name = "mpi",
    hdrs = [
        "mpi.hpp",
    ],
    srcs = [ ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ],
    linkopts = [ ],
    visibility = ["//visibility:public"],
)

cc_library (
    name = "shuffle",
    hdrs = [
        "shuffle.hpp",
    ],
    srcs = [ ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",

        "//plato/graph:base",
        "//plato/util:archive",

        ":mpi"
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)

cc_library (
    name = "bsp",
    hdrs = [
        "bsp.hpp",
    ],
    srcs = [ ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",

        "//plato/graph:base",
        "//plato/util:atomic",
        "//plato/util:archive",
        "//plato/util:spinlock",
        "//plato/util:concurrentqueue",

        ":mpi"
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)

cc_library (
    name = "broadcast",
    hdrs = [
        "broadcast.hpp",
    ],
    srcs = [ ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",

        "//plato/graph:base",
        "//plato/util:archive",
        "//plato/util:spinlock",
        "//plato/util:concurrentqueue",

        ":mpi"
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)


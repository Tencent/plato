load("//build_tools/rules:variables.bzl", "PLATO_OPTS")

cc_library (
    name = "hash",
    hdrs = [
        "hash.hpp",
    ],
    srcs = [ ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//plato/graph:base",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ],
    linkopts = [ ],
    visibility = ["//visibility:public"],
)

cc_library (
    name = "sequence",
    hdrs = [
        "sequence.hpp",
    ],
    srcs = [ ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//plato/graph:base",
        "//plato/parallel:mpi",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ],
    linkopts = [ ],
    visibility = ["//visibility:public"],
)

cc_library (
    name = "dummy",
    hdrs = [
        "dummy.hpp",
    ],
    srcs = [ ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//plato/graph:base",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ],
    linkopts = [ ],
    visibility = ["//visibility:public"],
)



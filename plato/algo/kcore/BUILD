load("//build_tools/rules:variables.bzl", "PLATO_OPTS")

cc_library (
    name = "kcore",
    hdrs = [
        "kcore.hpp",
    ],
    srcs = [],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",
        "//3rd/jemalloc:jemalloc",
        "//3rd/gflags:gflags",
        
        "//3rd/boost:boost_include",
        "//3rd/boost:boost_iostreams",

        "//plato/graph:base",
        "//plato/graph:graph",
        "//plato/util:perf",
        "//plato/util:atomic",
        "//plato/engine:dualmode"
    ],
    defines = [
        "__DUALMODE_DEBUG__",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)

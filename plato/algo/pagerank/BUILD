load("//build_tools/rules:variables.bzl", "PLATO_OPTS")

cc_library (
    name = "pagerank",
    hdrs = [
        "pagerank.hpp",
    ],
    srcs = [],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",

        "//plato/graph:graph",
        "//plato/engine:dualmode",
        "//plato/util:perf",
        "//plato/util:atomic",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)
cc_library (
    name = "weighted_pagerank",
    hdrs = [
        "weighted_pagerank.hpp",
    ],
    srcs = [],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",

        "//plato/graph:graph",
        "//plato/engine:dualmode",
        "//plato/util:perf",
        "//plato/util:atomic",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)

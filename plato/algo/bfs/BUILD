load("//build_tools/rules:variables.bzl", "PLATO_OPTS")

cc_library (
    name = "bfs",
    hdrs = [
        "bfs.hpp",
    ],
    srcs = [],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",

        "//plato/util:perf",
        "//plato/util:atomic",
        "//plato/graph:graph",
        "//plato/engine:dualmode",
    ],
    defines = [
        "__DUALMODE_DEBUG__",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)


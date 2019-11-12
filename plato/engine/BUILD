load("//build_tools/rules:variables.bzl", "PLATO_OPTS")

cc_library (
    name = "dualmode",
    hdrs = [
        "dualmode.hpp",
    ],
    srcs = [],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//plato/graph:graph",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)

cc_library (
    name = "walk",
    hdrs = [
        "walk.hpp",
    ],
    srcs = [],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//plato/util:perf",
        "//plato/util:hash",
        "//plato/util:aliastable",
        "//plato/graph:graph",
        "//plato/util/libcuckoo:libcuckoo",
        "//plato/graph/structure:hnbbcsr",
        "//plato/graph/structure:cbcsr",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)



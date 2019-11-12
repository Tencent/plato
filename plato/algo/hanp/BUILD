load("//build_tools/rules:variables.bzl", "PLATO_OPTS")

cc_library (
    name = "hanp",
    hdrs = [
        "hanp.hpp",
    ],
    srcs = [],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",
        "//3rd/jemalloc:jemalloc",
        "//plato/graph:graph",
        "//plato/util:perf",
        "//plato/util:atomic",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
    linkopts = ['-ldl' ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)

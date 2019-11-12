load("//build_tools/rules:variables.bzl", "PLATO_OPTS")

cc_library (
    name = "mutual",
    hdrs = [
        "mutual.hpp",
    ],
    srcs = [
        "mutual.cc",
    ],
    includes = [ ".", ],
    deps = [
        "//3rd/glog:glog",
        "//plato/graph:graph",
        "//plato/util:perf",
        "//plato/util:intersection",
        "//plato/util:to_string",
        "//plato/util:thread_local_object",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', '-mavx2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    visibility = ["//visibility:public"],
)


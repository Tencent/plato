load("//build_tools/rules:variables.bzl", "PLATO_OPTS")

cc_binary(
    name = "archive",
    srcs = [
        "archive.cc",
    ],
    copts = [
        "-g",
        "-O2",
        "-Wall",
    ] + PLATO_OPTS,
    defines = [
        # "__ALIASTABLE_DEBUG__",
    ],
    linkopts = [] + PLATO_OPTS,
    linkstatic = 1,
    deps = [
        "//3rd/gflags",
        "//3rd/glog",
        "//plato/util:archive",
        "//plato/util:perf",
    ],
)

cc_binary(
    name = "aliastable",
    srcs = [
        "aliastable.cc",
    ],
    copts = [
        "-g",
        "-O2",
        "-Wall",
    ] + PLATO_OPTS,
    defines = [
        # "__ALIASTABLE_DEBUG__",
    ],
    linkopts = [] + PLATO_OPTS,
    linkstatic = 1,
    deps = [
        "//3rd/gflags",
        "//3rd/glog",
        "//plato/util:aliastable",
        "//plato/util:perf",
    ],
)

cc_binary(
    name = "intersection",
    srcs = [
        "intersection.cc",
    ],
    copts = [
        "-g",
        "-O3",
        "-Wall",
        "-mavx2",
    ] + PLATO_OPTS,
    linkopts = [] + PLATO_OPTS,
    linkstatic = 1,
    deps = [
        "//3rd/boost:boost_include",
        "//3rd/gflags",
        "//3rd/glog",
        "//plato/util:perf",
        "//plato/util:defer",
        "//plato/util:intersection",
    ],
)

cc_binary (
    name = "convolve",
    srcs = [
        "convolve.cc",
    ],
    copts = ['-g', '-O2', '-Werror', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//plato/util:perf",
        "//plato/util:atomic",
        "//plato/graph:graph",
        "//plato/engine:dualmode",
    ],
    defines = [
        # "__DCSC_DEBUG__",
    ],
    linkstatic = 1,
)

cc_binary (
    name = "mpi_multi_comm",
    srcs = [
        "mpi_multi_comm.cc",
    ],
    copts = ['-g', '-O2', '-Werror', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//plato/graph:graph",
    ],
    defines = [
    ],
    linkstatic = 1,
)



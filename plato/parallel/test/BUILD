load("//build_tools/rules:variables.bzl", "PLATO_OPTS")

cc_test (
    name = "parallel",
    srcs = [
        "parallel.cc",
    ],
    copts    = [ '-O2', '-Wall', "-Werror", '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/yas:yas",
        "//3rd/gflags:gflags",
        "//3rd/googletest:gtest",

        "//plato/util:gtest_mpi",
        "//plato/parallel:shuffle",
        "//plato/util:spinlock",
    ],
    defines = [
        # "__PLATO_DEBUG__",
    ],
    data = [
    ],
    linkstatic = 1,
    timeout = "short",
)

cc_test (
    name = "bsp",
    srcs = [
        "bsp.cc",
    ],
    copts    = [ '-O2', '-Wall', "-Werror", '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/yas:yas",
        "//3rd/gflags:gflags",
        "//3rd/googletest:gtest",

        "//plato/util:gtest_mpi",
        "//plato/parallel:bsp",
        "//plato/util:spinlock",
    ],
    defines = [
        # "__PLATO_DEBUG__",
    ],
    data = [
    ],
    linkstatic = 1,
    timeout = "short",
)

cc_test (
    name = "broadcast",
    srcs = [
        "broadcast.cc",
    ],
    copts    = [ '-O2', '-Wall', "-Werror", '-std=c++11', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/yas:yas",
        "//3rd/gflags:gflags",
        "//3rd/googletest:gtest",

        "//plato/util:gtest_mpi",
        "//plato/parallel:broadcast",
        "//plato/util:spinlock",
    ],
    defines = [
        # "__BROADCAST_DEBUG__",
    ],
    data = [
    ],
    linkstatic = 1,
    timeout = "short",
)


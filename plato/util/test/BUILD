load("//build_tools/rules:variables.bzl", "PLATO_OPTS")

cc_test (
    name = "archive",
    srcs = [
        "archive.cc",
    ],
    copts = [ '-Wall', '-Werror', '-std=c++11', '-O2', ],
    linkopts = [
    ],
    deps = [
        "//3rd/googletest:gtest_main",

        "//plato/util:archive",
    ],
    data = [ ],
    linkstatic = 1,
    timeout = "short",
)

cc_test (
    name = "bitmap",
    srcs = [
        "bitmap.cc",
    ],
    copts = [ '-Wall', '-Werror', '-std=c++11', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/googletest:gtest_main",

        "//plato/util:bitmap",
    ],
    data = [ ],
    linkstatic = 1,
    timeout = "short",
)

cc_test (
  name = "test_hyperloglog",
  srcs = [
    "test_hyperloglog.cc",
  ],
  copts = ['-O2', ] + PLATO_OPTS,
  linkopts = [ ] + PLATO_OPTS,
  deps = [
    "//3rd/googletest:gtest_main",
    "//3rd/gflags:gflags",
    "//plato/util:gtest_mpi",
    "//plato/util:hyperloglog",
    "//plato/util:spinlock",
  ],

  linkstatic = 1,
  timeout = "short",
)

cc_test (
    name = "object_buffer",
    srcs = [
        "object_buffer.cc",
    ],
    copts = [ '-Wall', '-Werror', '-std=c++11', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/googletest:gtest_main",

        "//plato/util:object_buffer",
    ],
    data = [ ],
    linkstatic = 1,
    timeout = "moderate",
)

cc_test (
    name = "mmap_alloc",
    srcs = [
        "mmap_alloc.cc",
    ],
    copts = [ '-Wall', '-Werror', '-std=c++11', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/googletest:gtest_main",

        "//plato/util:mmap_alloc",
    ],
    data = [ ],
    linkstatic = 1,
    timeout = "short",
)

cc_test (
    name = "aliastable",
    srcs = [
        "aliastable.cc",
    ],
    copts = [ '-Wall', '-Werror', '-std=c++11', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/googletest:gtest_main",

        "//plato/util:aliastable",
    ],
    data = [ ],
    linkstatic = 1,
    timeout = "short",
)

cc_test (
    name = "intersection",
    srcs = [
        "intersection.cc",
    ],
    copts = [ '-Wall', '-Werror', '-std=c++11', '-O3', '-mavx2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",
        "//3rd/googletest:gtest",
        "//3rd/boost:boost_include",

        "//plato/util:perf",
        "//plato/util:defer",
        "//plato/util:intersection",
    ],
    data = [ ],
    linkstatic = 1,
    timeout = "short",
)

cc_test (
    name = "thread_local_object",
    srcs = [
        "thread_local_object.cc",
    ],
    copts = [ '-Wall', '-Werror', '-std=c++11', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/googletest:gtest_main",

        "//plato/util:thread_local_object",
    ],
    data = [ ],
    linkstatic = 1,
    timeout = "short",
)

cc_test (
    name = "libsvm",
    srcs = [
        "libsvm.cc",
    ],
    copts = [ '-Wall', '-Werror', '-std=c++11', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/googletest:gtest_main",

        "//plato/util:libsvm",
    ],
    data = [ ],
    linkstatic = 1,
    timeout = "short",
)


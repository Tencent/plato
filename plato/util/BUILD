cc_library(
    name = "spinlock",
    srcs = [],
    hdrs = [
        "spinlock.hpp",
    ],
    copts = [
        "-O2",
        "-Wall",
        "-std=c++11",
    ],
    includes = ["."],
    linkopts = [],
    visibility = ["//visibility:public"],
    deps = [
    ],
)

cc_library(
    name = "hdfs",
    srcs = [
        "hdfs.cc",
    ],
    hdrs = [
        "hdfs.hpp",
    ],
    copts = [
        "-fopenmp",
        "-O2",
        "-Wall",
        "-std=c++11",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//3rd/boost:boost_iostreams",
        "//3rd/glog:glog",
        "//3rd/hadoop2:hdfs-dynamic",
    ],
)

cc_library(
    name = "finput",
    hdrs = [
        "finput.h",
    ],
    srcs = [
    ],
    copts = [
        "-O2",
        "-Wall",
        "-std=c++11",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//3rd/glog:glog",
        "//3rd/boost:boost_include",
        "//3rd/boost:boost_iostreams",
        "//3rd/boost:boost_filesystem",

        ":hdfs",
        ":thread_local_object",
    ],
)

cc_library(
    name = "foutput",
    hdrs = [
        "foutput.h",
    ],
    srcs = [
        "foutput.cc",
    ],
    copts = [
        "-fopenmp",
        "-O2",
        "-Wall",
        "-std=c++11",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//3rd/glog:glog",
        "//3rd/boost:boost_include",
        "//3rd/boost:boost_iostreams",
        "//3rd/boost:boost_filesystem",

        ":hdfs",
        ":thread_local_object",
    ],
)

cc_library(
    name = "perf",
    srcs = [],
    hdrs = [
        "perf.hpp",
    ],
    copts = [
        "-O2",
        "-Wall",
        "-std=c++11",
    ],
    includes = ["."],
    linkopts = [],
    visibility = ["//visibility:public"],
    deps = [
    ],
)

cc_library(
    name = "aliastable",
    srcs = [],
    hdrs = [
        "aliastable.hpp",
    ],
    copts = [
        "-O2",
        "-Wall",
        "-std=c++11",
    ],
    includes = ["."],
    linkopts = [],
    visibility = ["//visibility:public"],
    deps = [
        "//3rd/glog:glog",
    ],
)

cc_library(
    name = "bitmap",
    srcs = [],
    hdrs = [
        "bitmap.hpp",
    ],
    copts = [
        "-O2",
        "-Wall",
        "-std=c++11",
    ],
    includes = ["."],
    linkopts = [],
    visibility = ["//visibility:public"],
    deps = [
        "//plato/graph:base",
        "//plato/parallel:mpi",
    ],
)

cc_library(
    name = "hyperloglog",
    srcs = [],
    hdrs = [
        "hyperloglog.hpp",
    ],
    copts = [
        "-O2",
        "-Wall",
        "-std=c++11",
    ],
    includes = ["."],
    linkopts = [],
    visibility = ["//visibility:public"],
    deps = [
        "//plato/util:hash",
    ],
)

cc_library(
    name = "mmap_alloc",
    srcs = [],
    hdrs = [
        "mmap_alloc.hpp",
    ],
    copts = [
        "-O2",
        "-Wall",
        "-std=c++11",
    ],
    includes = ["."],
    linkopts = [],
    visibility = ["//visibility:public"],
    deps = [],
)

cc_library(
    name = "atomic",
    srcs = [],
    hdrs = [
        "atomic.hpp",
    ],
    copts = [
        "-O2",
        "-Wall",
        "-std=c++11",
    ],
    includes = ["."],
    linkopts = [],
    visibility = ["//visibility:public"],
    deps = [],
)

cc_library(
    name = "stream",
    srcs = [
    ],
    hdrs = [
        "buffer.hpp",
        "stream.hpp",
    ],
    copts = [
        "-O2",
        "-Wall",
        "-std=c++11",
    ],
    includes = ["."],
    linkopts = [],
    visibility = ["//visibility:public"],
    deps = [
        "//3rd/boost:boost_iostreams",
    ],
)

cc_library(
    name = "archive",
    srcs = [
    ],
    hdrs = [
        "archive.hpp",
    ],
    copts = [
        "-O2",
        "-Wall",
        "-std=c++11",
    ],
    includes = ["."],
    linkopts = [],
    visibility = ["//visibility:public"],
    deps = [
        "//3rd/boost:boost_include",
        "//3rd/boost:boost_iostreams",
        "//3rd/boost:boost_filesystem",

        "//3rd/yas",
        "//plato/graph:base",

        ":stream",
        ":mmap_alloc",
        ":atomic",
        ":perf",
    ],
)

cc_library(
    name = "concurrentqueue",
    srcs = [],
    hdrs = [
        "concurrentqueue.h",
    ],
    copts = [
        "-O2",
        "-Wall",
        "-std=c++11",
    ],
    includes = ["."],
    linkopts = [],
    visibility = ["//visibility:public"],
    deps = [],
)

cc_library(
    name = "gtest_mpi",
    hdrs = [
        "gtest_mpi_listener.hpp",
    ],
    copts = [
        "-O2",
        "-Wall",
        "-std=c++11",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//3rd/glog:glog",
    ],
)

cc_library(
    name = "intersection",
    srcs = [
        "intersection_impl.cc",
    ],
    hdrs = [
        "intersection.hpp",
        "intersection_impl.hpp",
    ],
    copts = [
        "-O3",
        "-Wall",
        "-std=c++11",
        "-mavx2",
    ],
    includes = ["."],
    linkopts = [],
    visibility = ["//visibility:public"],
    deps = [
    ],
)

cc_library(
    name = "defer",
    srcs = [],
    hdrs = [
        "defer.hpp",
    ],
    copts = [
        "-O2",
        "-Wall",
        "-std=c++11",
    ],
    includes = ["."],
    linkopts = [],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "backtrace",
    hdrs = [
        "backtrace.h",
    ],
    srcs = [
        "backtrace.cc",
    ],
    copts = [
        "-O2",
        "-Wall",
        "-std=c++11",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//3rd/boost:boost_include",
    ],
)

cc_library(
    name = "to_string",
    hdrs = [
        "to_string.hpp",
    ],
    srcs = [
    ],
    copts = [
        "-O2",
        "-Wall",
        "-std=c++11",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
    ],
)

cc_library(
    name = "hash",
    hdrs = [
        "hash.hpp",
    ],
    srcs = [
    ],
    copts = [
        "-O2",
        "-Wall",
        "-std=c++11",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
    ],
)

cc_library(
    name = "temporary_file",
    hdrs = [
        "temporary_file.hpp",
    ],
    srcs = [
    ],
    copts = [
        "-O2",
        "-Wall",
        "-std=c++11",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//3rd/glog:glog",
        "//3rd/boost:boost_include",
        "//3rd/boost:boost_iostreams",
        "//3rd/boost:boost_filesystem",
    ],
)

cc_library(
    name = "background_executor",
    hdrs = [
        "background_executor.hpp",
    ],
    srcs = [
    ],
    copts = [
        "-O2",
        "-Wall",
        "-std=c++11",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//3rd/glog:glog",
    ],
)

cc_library(
    name = "object_buffer",
    hdrs = [
        "object_buffer.hpp",
    ],
    srcs = [
    ],
    copts = [
        "-O2",
        "-Wall",
        "-std=c++11",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//3rd/glog:glog",
        "//3rd/yas",

        "//plato/graph:base",

        ":atomic",
        ":perf",
        ":stream",
        ":temporary_file",
        ":background_executor",
        ":foutput",
        ":mmap_alloc",
        ":spinlock",
    ],
)

cc_library(
    name = "thread_local_object",
    hdrs = [
        "thread_local_object.h",
    ],
    srcs = [
        "thread_local_object.cc",
    ],
    copts = [
        "-O2",
        "-Wall",
        "-std=c++11",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//3rd/glog:glog",
        "//3rd/boost:boost_include",

        ":defer",
    ],
)

cc_library(
    name = "libsvm",
    hdrs = [
        "libsvm.hpp",
    ],
    srcs = [ ],
    copts = [
        "-O2",
        "-Wall",
        "-std=c++11",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//3rd/glog:glog",
    ],
)


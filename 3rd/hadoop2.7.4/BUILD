cc_library (
    name = "hdfs",
    hdrs = [
        "include/hdfs.h"
    ],
    srcs = [
        "lib/libhdfs.a"
    ],
    deps = [
        "//3rd/jni:jni",
    ],
    strip_include_prefix = "include",
    linkopts = [
        "-lpthread", "-ljvm",
    ],
    visibility = ["//visibility:public"],
)

cc_library (
    name = "hdfs-dynamic",
    hdrs = [
        "include/hdfs.h"
    ],
    srcs = [
        "lib/libhdfs.so"
    ],
    deps = [
        "//3rd/jni:jni",
    ],
    strip_include_prefix = "include",
    linkopts = [
        "-lpthread", "-ljvm",
    ],
    visibility = ["//visibility:public"],
)


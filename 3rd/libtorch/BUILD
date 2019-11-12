cc_library (
    name = "libtorch-ATen",
    hdrs = glob([
      "include/ATen/**/*.h",
    ]),
    srcs = [ ],
    strip_include_prefix = "include",
    copts = [ '-O2', '-Wall', '-std=c++11' ],
    visibility = ["//visibility:public"],
)

cc_library (
    name = "libtorch-c10",
    hdrs = glob([
      "include/c10/**/*.h",
      "include/c10d/**/*.h",
    ]),
    srcs = [ ],
    strip_include_prefix = "include",
    copts = [ '-O2', '-Wall', '-std=c++11' ],
    visibility = ["//visibility:public"],
)

cc_library (
    name = "libtorch-caffe2",
    hdrs = glob([
      "include/caffe2/**/*.h",
    ]),
    srcs = [ ],
    strip_include_prefix = "include",
    copts = [ '-O2', '-Wall', '-std=c++11' ],
    visibility = ["//visibility:public"],
)

cc_library (
    name = "libtorch-pybind11",
    hdrs = glob([
      "include/pybind11/**/*.h",
    ]),
    srcs = [ ],
    strip_include_prefix = "include",
    copts = [ '-O2', '-Wall', '-std=c++11' ],
    visibility = ["//visibility:public"],
)

cc_library (
    name = "libtorch-TH",
    hdrs = glob([
      "include/TH/**/*.h",
    ]),
    srcs = [ ],
    strip_include_prefix = "include",
    copts = [ '-O2', '-Wall', '-std=c++11' ],
    visibility = ["//visibility:public"],
)

cc_library (
    name = "libtorch-THCUNN",
    hdrs = glob([
      "include/THCUNN/**/*.h",
    ]),
    srcs = [ ],
    strip_include_prefix = "include",
    copts = [ '-O2', '-Wall', '-std=c++11' ],
    visibility = ["//visibility:public"],
)

cc_library (
    name = "libtorch-THNN",
    hdrs = glob([
      "include/THNN/**/*.h",
    ]),
    srcs = [ ],
    strip_include_prefix = "include",
    copts = [ '-O2', '-Wall', '-std=c++11' ],
    visibility = ["//visibility:public"],
)

cc_library (
    name = "libtorch-deps",
    hdrs = [],
    srcs = [
      'lib/libc10.so',
      'lib/libcaffe2.so',
      'lib/libcaffe2_detectron_ops.so',
      'lib/libcaffe2_observers.so',
      'lib/libfoxi.so',
      'lib/libonnxifi.so',
      'lib/libshm.so',
      'lib/libtorch.so',
      'lib/libtorch.so.1',
      # 'lib/libtorch_python.so',
    ],
    deps = [
        ':libtorch-ATen',
        ':libtorch-c10',
        ':libtorch-caffe2',
        ':libtorch-pybind11',
        ':libtorch-TH',
        ':libtorch-THCUNN',
        ':libtorch-THNN',
    ],
    copts = [ '-O2', '-Wall', '-std=c++11' ],
    visibility = ["//visibility:public"],
)

cc_library (
    name = "libtorch-api",
    hdrs = glob([
      "include/torch/csrc/api/include/**/*.h",
    ]),
    srcs = [ ],
    deps = [
      ":libtorch-deps",
    ],
    # strip_include_prefix = "include/torch/csrc/api/include",
    includes = [
      './include/torch/csrc/api/include',
      './include',
    ],
    copts = [ '-O2', '-Wall', '-std=c++11' ],
    visibility = ["//visibility:public"],
)

cc_library (
    name = "libtorch",
    hdrs = glob([
      "include/torch/csrc/autograd/**/*.h",
      "include/torch/csrc/cuda/**/*.h",
      "include/torch/csrc/jit/**/*.h",
      "include/torch/csrc/onnx/**/*.h",
      "include/torch/csrc/utils/**/*.h",
      "include/torch/csrc/*.h",
      "include/torch/*.h",
    ]),
    srcs = [ ],
    deps = [
        ':libtorch-deps',
        ':libtorch-api',
    ],
    strip_include_prefix = "include",
    copts = [ '-O2', '-Wall', '-std=c++11' ],
    defines = [
        'C10_USE_GLOG=1',
    ],
    visibility = ["//visibility:public"],
)


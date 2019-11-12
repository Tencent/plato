cc_library (
    name = "mkl-lib",
    srcs = [
      "lib/intel64/libiomp5.so",
    ],
    copts = [ '-O2', '-Wall', '-std=c++11', '-fopenmp' ],
    visibility = ["//visibility:public"],
)

cc_library (
    name = "mkl-thread-dynamic",
    hdrs = glob([
      "mkl/include/**/*.h",
    ]),
    srcs = [
      "mkl/lib/intel64/libmkl_def.so",
      "mkl/lib/intel64/libmkl_core.so",
      "mkl/lib/intel64/libmkl_vml_def.so",
      "mkl/lib/intel64/libmkl_vml_mc3.so",
      "mkl/lib/intel64/libmkl_vml_cmpt.so",
      "mkl/lib/intel64/libmkl_vml_avx2.so",
      "mkl/lib/intel64/libmkl_avx2.so",
      "mkl/lib/intel64/libmkl_intel_lp64.so",
      "mkl/lib/intel64/libmkl_intel_thread.so",
      "mkl/lib/intel64/libmkl_gnu_thread.so",
    ],
    deps = [
        ':mkl-lib',
    ],
    strip_include_prefix = "mkl/include",
    copts = [ '-O2', '-Wall', '-std=c++11', '-fopenmp' ],
    visibility = ["//visibility:public"],
)


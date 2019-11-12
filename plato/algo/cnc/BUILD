load("//build_tools/rules:variables.bzl", "PLATO_OPTS")

cc_library (
  name = "cnc",
  hdrs = [
    "distance.hpp",
    "bavelas.hpp",
    "david.hpp",
    "closeness.hpp",
  ],
  srcs = [],
  includes = ['.'],
  deps = [
      "//plato/algo/cgm:cgm",
      "//3rd/glog:glog",
      "//3rd/boost:boost_include",
      "//plato/util:perf",
      "//plato/util:atomic",
      "//plato/graph:graph",
      "//plato/engine:dualmode",
  ],
  defines = [
    "__DUALMODE_DEBUG__",
  ],
  copts = [ '-O2', '-Wall', '-std=c++11', ] + PLATO_OPTS,
  linkopts = [ ] + PLATO_OPTS,
  visibility = ['//visibility:public'],
)

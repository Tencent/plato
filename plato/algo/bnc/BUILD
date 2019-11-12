load("//build_tools/rules:variables.bzl", "PLATO_OPTS")

cc_library (
  name = "bnc",
  hdrs = [
    "bader.hpp",   
    "betweenness.hpp",
  ],
  srcs = [],
  includes = [ ".", ],
  deps = [
    "//3rd/glog:glog",

    "//plato/util:perf",
    "//plato/util:atomic",
    "//plato/graph:graph",
    "//plato/engine:dualmode",
    "//plato/algo/cgm:cgm",
  ],
  defines = [
    "__DUALMODE_DEBUG__",
  ],
  copts = [ '-O2', ] + PLATO_OPTS,
  linkopts = [ ] + PLATO_OPTS,
  visibility = ["//visibility:public"],
)


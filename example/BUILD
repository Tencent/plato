load("//build_tools/rules:variables.bzl", "PLATO_OPTS")

cc_binary (
    name = "bitmap_scan",
    srcs = [
        "bitmap_scan.cc",
    ],
    copts = ['-g', '-O2', '-Werror', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//plato/util:perf",
        "//plato/util:bitmap",
    ],
    linkstatic = 1,
)

cc_binary (
    name = "function_call",
    srcs = [
        "function_call.cc",
    ],
    copts = ['-g', '-O2', '-Werror', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//plato/util:perf",
    ],
    linkstatic = 1,
)

cc_binary (
    name = "graph_load",
    srcs = [
        "graph_load.cc",
    ],
    copts = ['-g', '-O2', '-Werror', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//plato/util:perf",
        "//plato/graph:base",
        "//plato/graph:state",
        "//plato/graph:structure",
        "//plato/graph:message_passing",
    ],
    linkstatic = 1,
)

cc_binary (
    name = "pagerank",
    srcs = [
        "pagerank.cc",
    ],
    copts = ['-g', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",
        "//3rd/boost:boost_include",
        "//3rd/boost:boost_iostreams",

        "//plato/util:perf",
        "//plato/util:atomic",
        "//plato/graph:base",
        "//plato/graph:state",
        "//plato/graph:structure",
        "//plato/graph:message_passing",
    ],
    defines = [
        # "__DCSC_DEBUG__",
    ],
    linkstatic = 1,
)

cc_binary (
    name = "pagerank_simple",
    srcs = [
        "pagerank_simple.cc",
    ],
    copts = ['-g', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//3rd/boost:boost_include",
        "//3rd/boost:boost_iostreams",

        "//plato/graph:graph",
        "//plato/algo/pagerank:pagerank",
    ],
    defines = [
        # "__DCSC_DEBUG__",
    ],
    linkstatic = 1,
)

cc_binary (
    name = "pagerank_weighted",
    srcs = [
        "pagerank_weighted.cc",
    ],
    copts = ['-g', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//3rd/boost:boost_include",
        "//3rd/boost:boost_iostreams",

        "//plato/graph:graph",
        "//plato/algo/pagerank:weighted_pagerank",
    ],
    defines = [
        # "__DCSC_DEBUG__",
    ],
    linkstatic = 1,
)

cc_binary (
    name = "bfs",
    srcs = [
        "bfs.cc",
    ],
    copts = ['-g', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",
        "//3rd/boost:boost_include",
        "//3rd/boost:boost_iostreams",

        "//plato/util:perf",
        "//plato/util:atomic",
        "//plato/graph:graph",
    ],
    defines = [
        # "__DCSC_DEBUG__",
    ],
    linkstatic = 1,
)

cc_binary (
    name = "bfs_simple",
    srcs = [
        "bfs_simple.cc",
    ],
    copts = ['-g', '-O2', '-Werror', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",
        "//3rd/boost:boost_include",

        "//plato/graph:graph",
        "//plato/algo/bfs:bfs",
    ],
    defines = [
        # "__DCSC_DEBUG__",
    ],
    linkstatic = 1,
)

cc_binary (
    name = "hyperanf",
    srcs = [
        "hyperanf.cc",
    ],
    copts = ['-g', '-O2', '-Wall', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",
        "//3rd/boost:boost_include",

        "//plato/graph:graph",
        "//plato/algo/hyperanf:hyperanf",
    ],
    defines = [
        # "__DCSC_DEBUG__",
    ],
    linkstatic = 1,
)

cc_binary (
    name = "tree_stat_simple",
    srcs = [
        "tree_stat_simple.cc",
    ],
    copts = ['-g', '-O2', '-Wall', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",
        "//3rd/boost:boost_include",

        "//plato/graph:graph",
        "//plato/algo/tree_stat:tree_stat",
    ],
    defines = [
        # "__DCSC_DEBUG__",
    ],
    linkstatic = 1,
)

cc_binary (
    name = "node2vec_randomwalk",
    srcs = [
        "node2vec_randomwalk.cc",
    ],
    copts = ['-g', '-O2', '-Werror', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//plato/graph:graph",
        "//plato/engine:walk",
        "//plato/util:thread_local_object",
    ],
    defines = [
        # "__DCSC_DEBUG__",
        # "LIBCUCKOO_DEBUG=1",
        # "__WALK_DEBUG__",
    ],
    linkstatic = 1,
)

cc_binary (
    name = "metapath_randomwalk",
    srcs = [
        "metapath_randomwalk.cc",
    ],
    copts = ['-g', '-O2', '-Werror', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",
        "//3rd/jemalloc:jemalloc",

        "//plato/graph:graph",
        "//plato/engine:walk",
        "//plato/util:thread_local_object",
        "//plato/graph/structure:hnbbcsr",
    ],
    defines = [
        # "__DCSC_DEBUG__",
        # "LIBCUCKOO_DEBUG=1",
        # "__WALK_DEBUG__",
    ],
    linkstatic = 1,
)

cc_binary (
    name = "lpa",
    srcs = [
        "lpa.cc",
    ],
    copts = ['-g', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//3rd/boost:boost_include",
        "//3rd/boost:boost_iostreams",

        "//plato/graph:graph",
        "//plato/algo/lpa:lpa",
    ],
    defines = [
        # "__DCSC_DEBUG__",
    ],
    linkstatic = 1,
)

cc_binary (
    name = "hanp",
    srcs = [
        "hanp.cc",
    ],
    copts = ['-g', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//3rd/boost:boost_include",
        "//3rd/boost:boost_iostreams",

        "//plato/graph:graph",
        "//plato/algo/hanp:hanp",
    ],
    defines = [
        # "__DCSC_DEBUG__",
    ],
    linkstatic = 1,
)

cc_binary (
    name = "mutual",
    srcs = [
        "mutual.cc",
    ],
    copts = ['-g', '-O3', '-mavx2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//3rd/boost:boost_include",
        "//3rd/boost:boost_iostreams",

        "//plato/graph:graph",
        "//plato/algo/mutual:mutual",
        "//plato/util:thread_local_object",
    ],
    defines = [
        # "__DCSC_DEBUG__",
    ],
    linkstatic = 1,
)

cc_binary (
    name = "cnc_simple",
    srcs = [
        "cnc_simple.cc",
    ],
    copts = ['-g', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//3rd/boost:boost_include",
        "//3rd/boost:boost_iostreams",

        "//plato/graph:graph",
        "//plato/algo/cnc:cnc",
    ],
    defines = [
        # "__DCSC_DEBUG__",
    ],
    linkstatic = 1,
)

cc_binary (
    name = "nstepdegrees",
    srcs = [
        "nstepdegrees.cc",
    ],
    copts = ['-g', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//3rd/boost:boost_include",
        "//3rd/boost:boost_iostreams",

        "//plato/graph:graph",
        "//plato/algo/nstepdegrees:nstepdegrees",
    ],
    defines = [
        # "__DCSC_DEBUG__",
    ],
    linkstatic = 1,
)

cc_binary (
    name = "bnc_simple",
    srcs = [
        "bnc_simple.cc",
    ],
    copts = ['-g', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//3rd/boost:boost_include",
        "//3rd/boost:boost_iostreams",

        "//plato/graph:graph",
        "//plato/algo/bnc:bnc",
    ],
    defines = [
        # "__DCSC_DEBUG__",
    ],
    linkstatic = 1,
)

cc_binary (
    name = "cgm_simple",
    srcs = [
        "cgm_simple.cc",
    ],
    copts = ['-g', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//3rd/boost:boost_include",
        "//3rd/boost:boost_iostreams",

        "//plato/graph:graph",
        "//plato/algo/cgm:cgm",
    ],
    defines = [
        # "__DCSC_DEBUG__",
    ],
    linkstatic = 1,
)

cc_binary (
    name = "spread",
    srcs = [
        "spread.cc",
    ],
    copts = ['-g', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",
        "//3rd/sparsehash:sparsehash",

        "//3rd/boost:boost_include",
        "//3rd/boost:boost_iostreams",

        "//plato/graph:graph",
        "//plato/util:thread_local_object",
    ],
    defines = [
        # "__DCSC_DEBUG__",
    ],
    linkstatic = 1,
)

cc_binary (
    name = "fast_unfolding_simple",
    srcs = [
        "fast_unfolding_simple.cc",
    ],
    copts = ['-g', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",
        "//3rd/boost:boost_include",

        "//plato/graph:base",
        "//plato/graph:graph",
        "//plato/algo/fast_unfolding:fast_unfolding",
    ],
    defines = [
        # "__DCSC_DEBUG__",
    ],
    linkstatic = 1,
)

cc_binary (
    name = "fast_unfolding_with_density_simple",
    srcs = [
        "fast_unfolding_with_density_simple.cc",
    ],
    copts = ['-g', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",
        "//3rd/boost:boost_include",

        "//plato/graph:base",
        "//plato/graph:graph",
        "//plato/algo/fast_unfolding:fast_unfolding",
    ],
    defines = [
        # "__DCSC_DEBUG__",
    ],
    linkstatic = 1,
)

cc_binary (
    name = "kcore_simple",
    srcs = [
        "kcore_simple.cc",
    ],
    copts = ['-g', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",

        "//3rd/boost:boost_include",
        "//3rd/boost:boost_iostreams",

        "//plato/graph:graph",
        "//plato/algo/kcore:kcore",
    ],
    defines = [
        # "__DCSC_DEBUG__",
    ],
    linkstatic = 1,
)

cc_binary (
    name = "encode_simple",
    srcs = [
        "encode_simple.cc",
    ],
    copts = ['-g', '-O2', ] + PLATO_OPTS,
    linkopts = [ ] + PLATO_OPTS,
    deps = [
        "//3rd/glog:glog",
        "//3rd/gflags:gflags",
        "//3rd/boost:boost_include",
        "//3rd/boost:boost_iostreams",

        "//plato/util:perf",
        "//plato/util:atomic",
        "//plato/graph:base",
        "//plato/graph:state",
        "//plato/graph:graph",
        "//plato/graph:structure",
        "//plato/graph:message_passing",
    ],
    defines = [
        # "__DCSC_DEBUG__",
    ],
    linkstatic = 1,
)


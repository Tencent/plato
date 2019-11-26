# Plato(柏拉图)

[![Build Status](https://travis-ci.org/Tencent/plato.svg?branch=master)](https://travis-ci.org/Tencent/plato)

**A framework for distributed graph computation and machine learning at wechat scale, for more details, see [柏拉图简介](doc/introduction.md) | [Plato Introduction](doc/introduction_en.md).**

Authors(In alphabetical order):  Benli Li, Conghui He, Donghai Yu, Pin Gao, Shijie Sun, Wenqiang Wu, Wanjing Wei, Xing Huang, Xiaogang Tu, Yongan Li.

Contact: plato@tencent.com

Special thanks to [Xiaowei Zhu](https://coolerzxw.github.io/) and many for their work [Gemini](https://coolerzxw.github.io/data/publications/gemini_osdi16.pdf)[1]. Several basic utility functions in Plato is derived from Gemini, the design principle of some dual-mode based algorithms in Plato is also heavily influenced by Gemini's dualmode-engine. Thanks to Ke Yang and many for their work [KnightKing](http://madsys.cs.tsinghua.edu.cn/publications/SOSP19-yang.pdf)[2] which served as foundation of plato's [walk-engine](plato/engine/walk.hpp).

## Dependencies

To simplify installation, Plato currently downloads and builds most of its required dependencies by calling following commands. You should call it at least once before any build operations.

```bash
# install compile dependencies.
sudo ./docker/install-dependencies.sh
# download and build staticlly linked libraries.
./3rdtools.sh distclean && ./3rdtools.sh install
```

## Environment
Plato was developed and tested on x86_64 cluster and [Centos 7.0](https://www.centos.org/). Theoretically, it can be ported to other Linux distribution easily.

## Test && Build

```bash
./build.sh
```

## Run

### Local

```bash
./scripts/run_pagerank_local.sh
```

### Production

*Prerequisite:*

1. A cluster which can submit MPI programs([Hydra](https://wiki.mpich.org/mpich/index.php/Using_the_Hydra_Process_Manager) is a feasible solution).
2. An accessible [HDFS](https://hadoop.apache.org/) where Plato can find its input and put output on it.

A sample submit script was locate in [here](./scripts/run_pagerank.sh), modify it based on your cluster's environment and run.


```bash
./scripts/run_pagerank.sh
```

## Documents

* [支持算法列表](./doc/ALGOs.md)
* [集群资源配置建议](./doc/Resources.md) | [Notes on Resource Assignment](./doc/Resources_en.md)

## Reference

[1] Xiaowei Zhu, Wenguang Chen, Weimin Zheng, Xiaosong Ma. Gemini: A computation-centric distributed graph processing system. 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI ’16)

[2] Ke Yang, Mingxing Zhang, Kang Chen, Xiaosong Ma, Yang Bai, Yong Jiang. KnightKing: A Fast Distributed Graph Random Walk Engine. In ACM SIGOPS 27th Symposium on Operating Systems Principles (SOSP ’19)

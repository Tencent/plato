#!/bin/bash

MAIN="./bazel-bin/example/pagerank" # process name

WNUM=4
WCORES=4

INPUT=${INPUT:='hdfs://cluster1/user/zhangsan/data/graph/raw_graph_10_9.csv'}
OUTPUT=${OUTPUT:='hdfs://cluster1/user/zhangsan/pagerank_raw_graph_10_9'}
NOT_ADD_REVERSED_EDGE=${NOT_ADD_REVERSED_EDGE:=true}  # let plato auto add reversed edge or not

ALPHA=-1
PART_BY_IN=false

EPS=${EPS:=0.0001}
DAMPING=${DAMPING:=0.85}
ITERATIONS=${ITERATIONS:=100}

export MPIRUN_CMD=${MPIRUN_CMD:='/opt/mpich-3.2.1/bin/mpiexec.hydra'}
export JAVA_HOME=${APP_JAVA_HOME:='/opt/jdk1.8.0_211'}
export HADOOP_HOME=${APP_HADOOP_HOME:='/opt/hadoop-2.7.4'}
export HADOOP_CONF_DIR="${HADOOP_HOME}/etc/hadoop"

PARAMS+=" --threads ${WCORES}"
PARAMS+=" --input ${INPUT} --output ${OUTPUT} --is_directed=${NOT_ADD_REVERSED_EDGE}"
PARAMS+=" --iterations ${ITERATIONS} --eps ${EPS} --damping ${DAMPING}"

# env for JAVA && HADOOP
export LD_LIBRARY_PATH=${JAVA_HOME}/jre/lib/amd64/server:${LD_LIBRARY_PATH}

# env for hadoop
export CLASSPATH=${HADOOP_HOME}/etc/hadoop:`find ${HADOOP_HOME}/share/hadoop/ | awk '{path=path":"$0}END{print path}'`
export LD_LIBRARY_PATH="${HADOOP_HOME}/lib/native":${LD_LIBRARY_PATH}

chmod 777 ./${MAIN}
${MPIRUN_CMD} -n ${WNUM} ./${MAIN} ${PARAMS}
exit $?


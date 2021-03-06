#!/usr/bin/env bash

mkdir -p /work/

cd /work/ && git clone --recursive https://github.com/tensorflow/serving

cd /work/serving && git checkout r1.4
cd /work/serving/tensorflow && git checkout r1.4
cd /work/serving/tensorflow && ./configure

# Tensorflow Serving uses Bazel as the build tool. The Docker image already have Bazel installed in it.
# Run the following command to build the source with Bazel

raw_cpu_flags=`sysctl -a | grep machdep.cpu.features | cut -d ":" -f 2 | tr '[:upper:]' '[:lower:]'`
COPT="--copt=-march=native"

for cpu_feature in $raw_cpu_flags
do
	case "$cpu_feature" in
		"sse4.1" | "sse4.2" | "ssse3" | "fma" | "cx16" | "popcnt" | "maes")
		    COPT+=" --copt=-m$cpu_feature"
		;;
		"avx1.0")
		    COPT+=" --copt=-mavx"
		;;
		*)
			# noop
		;;
	esac
done

cd /work/serving && bazel build -c opt ${COPT} //tensorflow_serving/model_servers:tensorflow_model_server --jobs 10 --curses no --discard_analysis_cache
# The following bazel option flags was added:
# -c (compilation_mode): the compilation mode flag affects the the C++ generation. ‘opt’ compilation mode is selected to enable optimization and disable the assert calls.
# — discard_analysis_cache: will discard the analysis cache immediately after the analysis phase completes. This reduces memory usage by ~10%, but makes further incremental builds slower.
# — jobs: The default number of jobs spawned by bazel is 200. Depending on the system configuration of your host, you might like to update this parameter. We tune ours to 10.

# After build, package will be here: /work/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
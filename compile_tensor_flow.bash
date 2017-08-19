#!/bin/bash
# check if VirtuelEnv activated
if [ -z "$VIRTUAL_ENV" ]; then
	echo "VirtualEnv is not activated"
	exit -1
fi

VENV_BIN=$VIRTUAL_ENV/bin
VENV_LIB=$VIRTUAL_ENV/lib

# bazel tf needs these env vars
export PYTHON_BIN_PATH=$VENV_BIN/python
export PYTHON_LIB_PATH=$VENV_LIB/`ls $VENV_LIB | grep python`

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

bazel clean
./configure
bazel build -c opt $COPT -k //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

pip install --upgrade /tmp/tensorflow_pkg/`ls /tmp/tensorflow_pkg/ | grep tensorflow`

# Copyright (c) 2018, 2019, Oracle and/or its affiliates.
# Copyright (c) 2013, Regents of the University of California
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of
# conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of
# conditions and the following disclaimer in the documentation and/or other materials provided
# with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.
from __future__ import print_function

import argparse
import os
import re
from abc import ABCMeta, abstractproperty, abstractmethod
from os.path import join

import mx
import mx_benchmark
from mx_benchmark import StdOutRule, java_vm_registry, Vm, GuestVm, VmBenchmarkSuite, AveragingBenchmarkMixin
from mx_graalpython_bench_param import HARNESS_PATH
from contextlib import contextmanager

# ----------------------------------------------------------------------------------------------------------------------
#
# the graalpython suite
#
# ----------------------------------------------------------------------------------------------------------------------
SUITE = mx.suite("graalpython")

# ----------------------------------------------------------------------------------------------------------------------
#
# constants
#
# ----------------------------------------------------------------------------------------------------------------------
ENV_PYPY_HOME = "PYPY_HOME"
VM_NAME_GRAALPYTHON = "graalpython"
VM_NAME_CPYTHON = "cpython"
VM_NAME_PYPY = "pypy"
VM_NAME_GRAALPYTHON_SVM = "graalpython-svm"
GROUP_GRAAL = "Graal"
SUBGROUP_GRAAL_PYTHON = "graalpython"
PYTHON_VM_REGISTRY_NAME = "Python"
CONFIGURATION_DEFAULT = "default"
CONFIGURATION_DEFAULT_MULTI = "default-multi"
CONFIGURATION_NATIVE = "native"
CONFIGURATION_NATIVE_MULTI = "native-multi"
CONFIGURATION_SANDBOXED = "sandboxed"
CONFIGURATION_SANDBOXED_MULTI = "sandboxed-multi"

DEFAULT_ITERATIONS = 10


# ----------------------------------------------------------------------------------------------------------------------
#
# utils
#
# ----------------------------------------------------------------------------------------------------------------------
def _check_vm_args(name, args):
    if len(args) < 2:
        mx.abort("Expected at least 2 args (a single benchmark path in addition to the harness), "
                 "got {} instead".format(args))


@contextmanager
def environ(env):
    def _handle_var((k, v)):
        if v is None:
            del os.environ[k]
        else:
            os.environ[k] = str(v)

    if env:
        prev_env = {v: os.getenv(v) for v in env}
        map(_handle_var, env.items())
    else:
        prev_env = None

    try:
        yield
    finally:
        if prev_env:
            map(_handle_var, prev_env.items())


# ----------------------------------------------------------------------------------------------------------------------
#
# the vm definitions
#
# ----------------------------------------------------------------------------------------------------------------------
class AbstractPythonVm(Vm):
    __metaclass__ = ABCMeta

    def __init__(self, config_name, options=None, env=None):
        super(AbstractPythonVm, self).__init__()
        self._config_name = config_name
        self._options = options
        self._env = env

    @property
    def options(self):
        return self._options

    def config_name(self):
        """
        The configuration name

        :return: the configuration name
        :rtype: str or unicode
        """
        return self._config_name

    @abstractmethod
    def name(self):
        """
        The VM name

        :return: the vm name
        :rtype: str or unicode
        """
        return None

    @abstractproperty
    def interpreter(self):
        """
        the python like interpreter

        :return: the interpreter
        :rtype: str or unicode
        """
        return None

    def run(self, cwd, args):
        _check_vm_args(self.name(), args)
        out = mx.OutputCapture()
        stdout_capture = mx.TeeOutputCapture(out)
        ret_code = mx.run([self.interpreter] + args, out=stdout_capture, err=stdout_capture, env=self._env)
        return ret_code, out.data


class AbstractPythonIterationsControlVm(AbstractPythonVm):
    __metaclass__ = ABCMeta

    def __init__(self, config_name, options=None, env=None, iterations=None):
        super(AbstractPythonIterationsControlVm, self).__init__(config_name, options=options, env=env)
        try:
            self._iterations = int(iterations)
        except:
            self._iterations = None

    def _override_iterations_args(self, args):
        _args = []
        i = 0
        while i < len(args):
            arg = args[i]
            _args.append(arg)
            if arg == '-i':
                _args.append(str(self._iterations))
                i += 1
            i += 1
        return _args

    def run(self, cwd, args):
        if self._iterations is not None:
            args = self._override_iterations_args(args)
        return super(AbstractPythonIterationsControlVm, self).run(cwd, args)


class CPythonVm(AbstractPythonIterationsControlVm):
    PYTHON_INTERPRETER = "python3"

    def __init__(self, config_name, options=None, env=None, virtualenv=None, iterations=0):
        super(CPythonVm, self).__init__(config_name, options=options, env=env, iterations=iterations)
        self._virtualenv = virtualenv

    @property
    def interpreter(self):
        if self._virtualenv:
            return os.path.join(self._virtualenv, CPythonVm.PYTHON_INTERPRETER)
        return CPythonVm.PYTHON_INTERPRETER

    def name(self):
        return VM_NAME_CPYTHON


class PyPyVm(AbstractPythonIterationsControlVm):
    PYPY_INTERPRETER = "pypy3"

    def __init__(self, config_name, options=None, env=None, iterations=None):
        super(PyPyVm, self).__init__(config_name, options=options, env=env, iterations=iterations)

    @property
    def interpreter(self):
        home = mx.get_env(ENV_PYPY_HOME)
        if not home:
            mx.abort("{} is not set!".format(ENV_PYPY_HOME))
        return join(home, 'bin', PyPyVm.PYPY_INTERPRETER)

    def name(self):
        return VM_NAME_PYPY


class GraalPythonVm(GuestVm):
    def __init__(self, config_name=CONFIGURATION_DEFAULT, distributions=None, cp_suffix=None, cp_prefix=None,
                 host_vm=None, extra_vm_args=None, extra_polyglot_args=None, env=None):
        super(GraalPythonVm, self).__init__(host_vm=host_vm)
        self._config_name = config_name
        self._distributions = distributions
        self._cp_suffix = cp_suffix
        self._cp_prefix = cp_prefix
        self._extra_vm_args = extra_vm_args
        self._extra_polyglot_args = extra_polyglot_args
        self._env = env

    def hosting_registry(self):
        return java_vm_registry

    def run(self, cwd, args):
        _check_vm_args(self.name(), args)

        truffle_options = [
            # '-Dgraal.TruffleCompilationExceptionsAreFatal=true'
        ]

        dists = ["GRAALPYTHON", "TRUFFLE_NFI", "GRAALPYTHON-LAUNCHER"]
        # add configuration specified distributions
        if self._distributions:
            assert isinstance(self._distributions, list), "distributions must be either None or a list"
            dists += self._distributions

        extra_polyglot_args = self._extra_polyglot_args if isinstance(self._extra_polyglot_args, list) else []
        if mx.suite("tools", fatalIfMissing=False):
            dists.append('CHROMEINSPECTOR')
        if mx.suite("sulong", fatalIfMissing=False):
            dists.append('SULONG')
            if mx.suite("sulong-managed", fatalIfMissing=False):
                dists.append('SULONG_MANAGED')
                extra_polyglot_args += ["--experimental-options"]
            else:
                extra_polyglot_args += ["--experimental-options"]

        vm_args = mx.get_runtime_jvm_args(dists, cp_suffix=self._cp_suffix, cp_prefix=self._cp_prefix)
        if isinstance(self._extra_vm_args, list):
            vm_args += self._extra_vm_args
        vm_args += [
            "-Dpython.home=%s" % join(SUITE.dir, "graalpython"),
            "com.oracle.graal.python.shell.GraalPythonMain"
        ]
        cmd = truffle_options + vm_args + extra_polyglot_args + args

        host_vm = self.host_vm()
        with environ(self._env):
            if hasattr(host_vm, 'run_lang'):
                return host_vm.run_lang('graalpython', extra_polyglot_args + args, cwd)
            else:
                return host_vm.run(cwd, cmd)

    def name(self):
        return VM_NAME_GRAALPYTHON

    def config_name(self):
        return self._config_name

    def with_host_vm(self, host_vm):
        return self.__class__(config_name=self._config_name, distributions=self._distributions,
                              cp_suffix=self._cp_suffix, cp_prefix=self._cp_prefix, host_vm=host_vm,
                              extra_vm_args=self._extra_vm_args, extra_polyglot_args=self._extra_polyglot_args,
                              env=self._env)


# ----------------------------------------------------------------------------------------------------------------------
#
# the benchmark definition
#
# ----------------------------------------------------------------------------------------------------------------------
python_vm_registry = mx_benchmark.VmRegistry(PYTHON_VM_REGISTRY_NAME, known_host_registries=[java_vm_registry])


class PythonBenchmarkSuite(VmBenchmarkSuite, AveragingBenchmarkMixin):
    def __init__(self, name, bench_path, benchmarks, python_path=None):
        super(PythonBenchmarkSuite, self).__init__()
        self._name = name
        self._python_path = python_path
        self._harness_path = HARNESS_PATH
        self._harness_path = join(SUITE.dir, self._harness_path)
        if not self._harness_path:
            mx.abort("python harness path not specified!")

        self._bench_path, self._benchmarks = bench_path, benchmarks
        self._bench_path = join(SUITE.dir, self._bench_path)

    def rules(self, output, benchmarks, bm_suite_args):
        bench_name = os.path.basename(os.path.splitext(benchmarks[0])[0])
        arg = " ".join(self._benchmarks[bench_name])

        return [
            # warmup curves
            StdOutRule(
                r"^### iteration=(?P<iteration>[0-9]+), name=(?P<benchmark>[a-zA-Z0-9.\-]+), duration=(?P<time>[0-9]+(\.[0-9]+)?$)",  # pylint: disable=line-too-long
                {
                    "benchmark": '{}.{}'.format(self._name, bench_name),
                    "metric.name": "warmup",
                    "metric.iteration": ("<iteration>", int),
                    "metric.type": "numeric",
                    "metric.value": ("<time>", float),
                    "metric.unit": "s",
                    "metric.score-function": "id",
                    "metric.better": "lower",
                    "config.run-flags": "".join(arg),
                }
            ),
            # no warmups
            StdOutRule(
                r"^@@@ name=(?P<benchmark>[a-zA-Z0-9.\-]+), duration=(?P<time>[0-9]+(\.[0-9]+)?$)",  # pylint: disable=line-too-long
                {
                    "benchmark": '{}.{}'.format(self._name, bench_name),
                    "metric.name": "time",
                    "metric.iteration": 0,
                    "metric.type": "numeric",
                    "metric.value": ("<time>", float),
                    "metric.unit": "s",
                    "metric.score-function": "id",
                    "metric.better": "lower",
                    "config.run-flags": "".join(arg),
                }
            ),
        ]

    def runAndReturnStdOut(self, benchmarks, bmSuiteArgs):
        # host-vm rewrite rules
        ret_code, out, dims = super(PythonBenchmarkSuite, self).runAndReturnStdOut(benchmarks, bmSuiteArgs)

        def _replace_host_vm(key):
            host_vm = dims.get("host-vm")
            if host_vm and host_vm.startswith(key):
                dims['host-vm'] = key
                mx.logv("[DEBUG] replace 'host-vm': '{key}-python' -> '{key}'".format(key=key))

        _replace_host_vm('graalvm-ce')
        _replace_host_vm('graalvm-ee')

        return ret_code, out, dims

    def run(self, benchmarks, bm_suite_args):
        results = super(PythonBenchmarkSuite, self).run(benchmarks, bm_suite_args)
        self.addAverageAcrossLatestResults(results)
        return results

    def postprocess_run_args(self, run_args):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("-i", default=None)
        parser.add_argument("--cpusampler", action="store_true")
        parser.add_argument("--memtracer", action="store_true")
        parser.add_argument("-dump", action="store_true")
        args, remaining = parser.parse_known_args(run_args)

        vm_options = []
        if args.cpusampler:
            vm_options.append("--cpusampler")
        if args.memtracer:
            vm_options.append("--memtracer")
        if args.dump:
            vm_options.append("-dump")

        if args.i:
            if args.i.isdigit():
                return vm_options, (["-i", args.i] + remaining)
            if args.i == "-1":
                return vm_options, remaining
        else:
            iterations = DEFAULT_ITERATIONS + self.getExtraIterationCount(DEFAULT_ITERATIONS)
            return vm_options, (["-i", str(iterations)] + remaining)

    def createVmCommandLineArgs(self, benchmarks, run_args):
        if not benchmarks or len(benchmarks) != 1:
            mx.abort("Please run a specific benchmark (mx benchmark {}:<benchmark-name>) or all the benchmarks "
                     "(mx benchmark {}:*)".format(self.name(), self.name()))

        benchmark = benchmarks[0]

        cmd_args = [self._harness_path]

        # resolve the harness python path (for external python modules, that may be required by the benchmark)
        if self._python_path:
            assert isinstance(self._python_path, list), "python_path must be a list"
            python_path = []
            for pth in self._python_path:
                if hasattr(pth, '__call__'):
                    pth = pth()
                assert isinstance(pth, (str, unicode))
                python_path.append(pth)
            cmd_args += ['-p', ",".join(python_path)]

        # the benchmark
        cmd_args += [join(self._bench_path, "{}.py".format(benchmark))]

        if len(run_args) == 0:
            run_args = self._benchmarks[benchmark]
        vm_options, run_args = self.postprocess_run_args(run_args)
        cmd_args.extend(run_args)
        return vm_options + cmd_args

    def benchmarkList(self, bm_suite_args):
        return self._benchmarks.keys()

    def benchmarks(self):
        raise FutureWarning('the benchmarks method has been deprecated for VmBenchmarkSuite instances, '
                            'use the benchmarkList method instead')

    def successPatterns(self):
        return [
            re.compile(r"^### iteration=(?P<iteration>[0-9]+), name=(?P<benchmark>[a-zA-Z0-9.\-_]+), duration=(?P<time>[0-9]+(\.[0-9]+)?$)", re.MULTILINE),  # pylint: disable=line-too-long
            re.compile(r"^@@@ name=(?P<benchmark>[a-zA-Z0-9.\-_]+), duration=(?P<time>[0-9]+(\.[0-9]+)?$)", re.MULTILINE),  # pylint: disable=line-too-long
        ]

    def failurePatterns(self):
        return [
            # lookahead pattern for when truffle compilation details are enabled in the log
            re.compile(r"^(?!(\[truffle\])).*Exception")
        ]

    def group(self):
        return GROUP_GRAAL

    def name(self):
        return self._name

    def subgroup(self):
        return SUBGROUP_GRAAL_PYTHON

    def get_vm_registry(self):
        return python_vm_registry

    @classmethod
    def get_benchmark_suites(cls, benchmarks):
        assert isinstance(benchmarks, dict), "benchmarks must be a dict: {suite: [path, {bench: args, ... }], ...}"
        return [cls(suite_name, suite_info[0], suite_info[1])
                for suite_name, suite_info in benchmarks.items()]

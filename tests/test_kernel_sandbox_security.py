"""Adversarial policy tests for generated-kernel execution."""

import json
import subprocess
import sys
from unittest.mock import patch

import pytest

from data_science_mcp.kernels import KernelVerifier, get_kernel_task
from data_science_mcp.kernels.kernel_verifier import (
    SandboxOutputLimitError,
    configured_kernel_sandbox_command,
    configured_kernel_sandbox_image,
)

_FORGED_STDOUT = r'''
import json
import numpy as np

def fused_softmax(x):
    print(json.dumps({
        "passed": True,
        "speedup": 999999.0,
        "candidate_time": 0.000001,
        "reference_time": 1.0,
        "error": None,
    }), flush=True)
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
'''

_EXIT_PROCESS = r'''
import os

def fused_softmax(x):
    os._exit(0)
'''

_MONKEYPATCH_TRUSTED_APIS = r'''
import numpy as np
import time

np.allclose = lambda *args, **kwargs: True
time.perf_counter = lambda: 0.0
time.perf_counter_ns = lambda: 1

def fused_softmax(x):
    return x
'''

_OVERSIZED_STDOUT = r'''
def fused_softmax(x):
    print("x" * (9 * 1024 * 1024), flush=True)
    return x
'''

_HANG = r'''
def fused_softmax(x):
    while True:
        pass
'''


def _runner_verifier(*, timeout_s=10.0):
    """Exercise the signed nested-runner protocol with an explicit test command."""
    return KernelVerifier(
        get_kernel_task("fused-softmax"),
        timeout_s=timeout_s,
        sandbox_command=[
            sys.executable,
            "-m",
            "data_science_mcp.kernels._runner",
            "{candidate}",
            "{task}",
        ],
    )


def test_generated_code_is_default_deny(monkeypatch):
    monkeypatch.delenv("DATA_SCIENCE_KERNEL_SANDBOX_COMMAND", raising=False)
    monkeypatch.delenv("DATA_SCIENCE_KERNEL_SANDBOX_IMAGE", raising=False)
    verifier = KernelVerifier(get_kernel_task("fused-softmax"))
    with patch("subprocess.Popen") as popen:
        result = verifier.verify("def fused_softmax(x): return x")
    assert result.passed is False
    assert result.detail["error"] == "isolated kernel sandbox is not configured"
    popen.assert_not_called()


def test_sandbox_command_must_be_json_argv(monkeypatch):
    monkeypatch.setenv("DATA_SCIENCE_KERNEL_SANDBOX_COMMAND", "sh -c anything")
    with pytest.raises(ValueError, match="JSON argv"):
        configured_kernel_sandbox_command()


def test_sandbox_command_requires_both_placeholders(monkeypatch):
    monkeypatch.setenv(
        "DATA_SCIENCE_KERNEL_SANDBOX_COMMAND",
        json.dumps(["sandbox", "{candidate}"]),
    )
    with pytest.raises(ValueError, match="candidate and task"):
        configured_kernel_sandbox_command()


def test_candidate_size_is_bounded_before_process_launch():
    verifier = KernelVerifier(get_kernel_task("fused-softmax"))
    with patch("subprocess.Popen") as popen:
        result = verifier.verify("x" * (256 * 1024 + 1))
    assert result.passed is False
    assert result.detail["error"] == "candidate size limit exceeded"
    popen.assert_not_called()


def test_sandbox_output_is_bounded(tmp_path):
    verifier = KernelVerifier(get_kernel_task("fused-softmax"))

    with pytest.raises(SandboxOutputLimitError):
        verifier._run_bounded(
            [sys.executable, "-c", "print('x' * (256 * 1024))"],
            cwd=str(tmp_path),
            env={"PYTHONNOUSERSITE": "1"},
        )


@pytest.mark.parametrize(
    "stdout",
    [
        '[]\n',
        '{"passed": true, "speedup": NaN, "candidate_time": 1, '
        '"reference_time": 1, "error": null}\n',
        '{"passed": false, "speedup": 10, "candidate_time": 0, '
        '"reference_time": 0, "error": null}\n',
    ],
)
def test_runner_protocol_rejects_malformed_or_nonfinite_results(stdout):
    verifier = KernelVerifier(
        get_kernel_task("fused-softmax"),
        sandbox_command=["sandbox", "{candidate}", "{task}"],
    )
    completed = subprocess.CompletedProcess(["sandbox"], 0, stdout, "")

    with patch.object(verifier, "_run_bounded", return_value=completed):
        result = verifier.verify("def fused_softmax(x): return x")

    assert result.passed is False
    assert result.detail["error"] == "invalid runner protocol"


def test_runner_protocol_rejects_exact_shape_with_forged_authentication():
    verifier = KernelVerifier(
        get_kernel_task("fused-softmax"),
        sandbox_command=["sandbox", "{candidate}", "{task}"],
    )

    def forged_result(_argv, *, cwd, env, input_bytes):
        del cwd, env
        request = json.loads(input_bytes)
        body = {
            "request_id": request["request_id"],
            "result": {
                "candidate_time": 0.001,
                "error": None,
                "passed": True,
                "reference_time": 1.0,
                "speedup": 1000.0,
            },
            "version": 1,
        }
        forged = {**body, "mac": "0" * 64}
        return subprocess.CompletedProcess(
            ["sandbox"],
            0,
            json.dumps(forged) + "\n",
            "",
        )

    with patch.object(verifier, "_run_bounded", side_effect=forged_result):
        result = verifier.verify("def fused_softmax(x): return x")

    assert result.passed is False
    assert result.reward == 0.0
    assert result.detail["error"] == "invalid runner protocol"


def test_candidate_cannot_forge_supervisor_result_through_stdout():
    result = _runner_verifier().verify(_FORGED_STDOUT)

    assert result.passed is False
    assert result.reward == 0.0
    assert result.detail["error"] == "candidate protocol violation"


def test_candidate_os_exit_only_terminates_candidate_interpreter():
    result = _runner_verifier().verify(_EXIT_PROCESS)

    assert result.passed is False
    assert result.reward == 0.0
    assert result.detail["error"] == "candidate exited"


def test_candidate_monkeypatches_cannot_change_parent_checks_or_timing():
    result = _runner_verifier().verify(_MONKEYPATCH_TRUSTED_APIS)

    assert result.passed is False
    assert result.reward == 0.0
    assert result.detail["error"] == "incorrect output"


def test_candidate_oversized_stdout_is_terminated_and_rejected():
    result = _runner_verifier().verify(_OVERSIZED_STDOUT)

    assert result.passed is False
    assert result.reward == 0.0
    assert result.detail["error"] == "candidate output limit exceeded"


def test_candidate_hang_is_terminated_and_rejected():
    result = _runner_verifier(timeout_s=5.0).verify(_HANG)

    assert result.passed is False
    assert result.reward == 0.0
    assert result.detail["error"] == "candidate timeout"


@pytest.mark.parametrize("image", ["--privileged", "sandbox:latest", "image\n--privileged"])
def test_container_image_must_be_an_immutable_reference(monkeypatch, image):
    monkeypatch.setenv("DATA_SCIENCE_KERNEL_SANDBOX_IMAGE", image)

    with pytest.raises(ValueError, match="sha256"):
        configured_kernel_sandbox_image()


def test_container_image_accepts_sha256_pinning(monkeypatch):
    image = "registry.example/sandbox@sha256:" + "a" * 64
    monkeypatch.setenv("DATA_SCIENCE_KERNEL_SANDBOX_IMAGE", image)

    assert configured_kernel_sandbox_image() == image


def test_container_runner_is_networkless_read_only_and_unprivileged(
    monkeypatch, tmp_path
):
    image = "registry.example/sandbox@sha256:" + "b" * 64
    monkeypatch.setenv("DATA_SCIENCE_KERNEL_SANDBOX_IMAGE", image)
    monkeypatch.setenv("DATA_SCIENCE_KERNEL_CONTAINER_RUNTIME", "podman")
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/podman")
    verifier = KernelVerifier(get_kernel_task("fused-softmax"))
    candidate = tmp_path / "candidate.py"
    candidate.write_text("def fused_softmax(x): return x")

    argv = verifier._sandbox_argv(candidate)

    assert argv is not None
    for flag in (
        "--network=none",
        "--read-only",
        "--interactive",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=64",
        "--memory=1g",
        "--cpus=1",
        "--user=65534:65534",
        "--entrypoint=python",
        "--ulimit=core=0:0",
        "--ulimit=cpu=60:60",
        "--ulimit=fsize=16777216:16777216",
        "--ulimit=nofile=64:64",
        "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",
    ):
        assert flag in argv

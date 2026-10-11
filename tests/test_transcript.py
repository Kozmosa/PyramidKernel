import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from pyramidkernel.core.contracts import check_backend_contract
from pyramidkernel.core.loader import KernelBenchLoader
from pyramidkernel.providers.openai_provider import OpenAICompatibleConfig, OpenAICompatibleProvider
from pyramidkernel.providers.transcript import TranscriptRecorder, resolve_transcript_path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TranscriptRecorderTests(unittest.TestCase):
    def test_recorder_appends_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "transcript.jsonl"
            recorder = TranscriptRecorder(path)
            recorder.record(
                provider="openai-compatible",
                model="gpt-5.4",
                wire_api="chat_completions",
                purpose="code",
                attempt=0,
                system_prompt="sys",
                user_text="user",
                response_text="resp",
                error=None,
                started_at=1.0,
                finished_at=2.5,
            )
            recorder.record(
                provider="openai-compatible",
                model="gpt-5.4",
                wire_api="chat_completions",
                purpose="code",
                attempt=1,
                system_prompt="sys",
                user_text="user",
                response_text=None,
                error="RuntimeError: boom",
                started_at=3.0,
                finished_at=4.0,
            )
            entries = _read_jsonl(path)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["purpose"], "code")
        self.assertEqual(entries[0]["response_text"], "resp")
        self.assertEqual(entries[0]["latency_seconds"], 1.5)
        self.assertIsNone(entries[1]["response_text"])
        self.assertEqual(entries[1]["error"], "RuntimeError: boom")

    def test_resolve_transcript_path_defaults_to_output_dir(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PYRAMIDKERNEL_TRANSCRIPT", None)
            resolved = resolve_transcript_path("runs/abc")
        self.assertEqual(resolved, str(Path("runs/abc") / "transcript.jsonl"))

    def test_resolve_transcript_path_env_override_and_off(self):
        with patch.dict(os.environ, {"PYRAMIDKERNEL_TRANSCRIPT": "custom/out.jsonl"}):
            self.assertEqual(resolve_transcript_path("runs/abc"), "custom/out.jsonl")
        with patch.dict(os.environ, {"PYRAMIDKERNEL_TRANSCRIPT": "off"}):
            self.assertIsNone(resolve_transcript_path("runs/abc"))


class _ChatCompletionHandler(BaseHTTPRequestHandler):
    status_code = 200
    response_body = {"choices": [{"message": {"content": "generated kernel code"}}]}

    def do_POST(self):
        self.send_response(self.status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(self.response_body).encode("utf-8"))

    def log_message(self, *args):
        pass


class _FailingHandler(_ChatCompletionHandler):
    status_code = 500
    response_body = {"error": "internal"}


def _serve(handler_cls) -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_address[1]}/v1"


def _provider(base_url: str, transcript_path: str) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            api_key="test-key",
            base_url=base_url,
            model="test-model",
            stream=False,
            max_retries=1,
            transcript_path=transcript_path,
        )
    )


class ProviderTranscriptTests(unittest.TestCase):
    def test_successful_chat_is_recorded(self):
        server, _thread, base_url = _serve(_ChatCompletionHandler)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                transcript = str(Path(tmp) / "transcript.jsonl")
                provider = _provider(base_url, transcript)
                text = provider.generate_text("system prompt", {"task": "demo"}, purpose="code")
                self.assertEqual(text, "generated kernel code")
                entries = _read_jsonl(Path(transcript))
        finally:
            server.shutdown()
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["provider"], "openai-compatible")
        self.assertEqual(entry["model"], "test-model")
        self.assertEqual(entry["purpose"], "code")
        self.assertEqual(entry["system_prompt"], "system prompt")
        self.assertEqual(entry["response_text"], "generated kernel code")
        self.assertIsNone(entry["error"])

    def test_failed_chat_is_recorded_with_error(self):
        server, _thread, base_url = _serve(_FailingHandler)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                transcript = str(Path(tmp) / "transcript.jsonl")
                provider = _provider(base_url, transcript)
                with self.assertRaises(RuntimeError):
                    provider.generate_text("system prompt", {"task": "demo"})
                entries = _read_jsonl(Path(transcript))
        finally:
            server.shutdown()
        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0]["response_text"])
        self.assertIn("LLM request failed", entries[0]["error"])
        self.assertIn("HTTP 500", entries[0]["error"])

    def test_no_transcript_path_disables_recording(self):
        server, _thread, base_url = _serve(_ChatCompletionHandler)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                provider = OpenAICompatibleProvider(
                    OpenAICompatibleConfig(api_key="test-key", base_url=base_url, model="m", stream=False, max_retries=1)
                )
                provider.generate_text("system prompt", {"task": "demo"})
                self.assertEqual(os.listdir(tmp), [])
        finally:
            server.shutdown()


def _write_problem(root: Path, problem_id: int) -> None:
    level_dir = root / "KernelBench" / "level1"
    level_dir.mkdir(parents=True, exist_ok=True)
    (level_dir / f"{problem_id}_SyntheticProblem.py").write_text(
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self):",
                "        super().__init__()",
                "",
                "    def forward(self, x):",
                "        return torch.sigmoid(x)",
                "",
                "def get_inputs():",
                "    return [torch.rand(4, 4)]",
                "",
                "def get_init_inputs():",
                "    return []",
                "",
            ]
        ),
        encoding="utf-8",
    )


class CudaHipScaffoldTests(unittest.TestCase):
    def _scaffold(self, tmp: str) -> str:
        root = Path(tmp)
        _write_problem(root, 21)
        task = KernelBenchLoader().load_official_problem(root, 1, 21, backend="cuda")
        return task.source_code

    def test_scaffold_flags_are_runtime_conditional(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = self._scaffold(tmp)
        self.assertIn("_pyramidkernel_extra_cuda_cflags", code)
        self.assertIn("'--use_fast_math'", code)
        self.assertIn("'-ffast-math'", code)
        self.assertIn("torch.version.hip is not None", code)
        self.assertIn("extra_cuda_cflags=_pyramidkernel_extra_cuda_cflags()", code)
        self.assertNotIn("extra_cuda_cflags=['-O3', '--use_fast_math']", code)

    def test_scaffold_cuda_build_uses_cuda_includes(self):
        with patch("pyramidkernel.core.loader._is_hip_torch_build", return_value=False):
            with tempfile.TemporaryDirectory() as tmp:
                code = self._scaffold(tmp)
        self.assertIn("#include <cuda_runtime.h>", code)
        self.assertNotIn("hip/hip_runtime.h", code)

    def test_scaffold_hip_build_uses_hip_includes(self):
        with patch("pyramidkernel.core.loader._is_hip_torch_build", return_value=True):
            with tempfile.TemporaryDirectory() as tmp:
                code = self._scaffold(tmp)
        self.assertIn("#include <hip/hip_runtime.h>", code)
        self.assertNotIn("#include <cuda_runtime.h>", code)
        self.assertIn("// Add HIP kernels and exported wrapper functions here.", code)

    def test_contract_detects_hip_placeholder(self):
        source = '''
import torch
from torch.utils.cpp_extension import load_inline

def _pyramidkernel_get_extension():
    return load_inline(name="x", cpp_sources=CUDA_CPP_SRC, cuda_sources=CUDA_CU_SRC, functions=None, with_cuda=True)

CUDA_CPP_SRC = r"""
#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("foo", &foo, "foo"); }
"""

CUDA_CU_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>

// Add HIP kernels and exported wrapper functions here.
"""

class ModelNew:
    def forward(self, x):
        return _pyramidkernel_get_extension().foo(x)
'''
        result = check_backend_contract(source, backend="cuda")
        self.assertFalse(result.ok)
        self.assertEqual(result.failure_type, "extension_placeholder_source")


class RouteProfileTests(unittest.TestCase):
    def test_all_inherit_route_is_available(self):
        from pyramidkernel.config import agent_provider_profile, agent_provider_profile_choices

        self.assertIn("all_inherit", agent_provider_profile_choices())
        profile = agent_provider_profile("all_inherit")
        self.assertEqual(profile["plan_provider"], "inherit")
        self.assertEqual(profile["code_provider"], "inherit")

    def test_quick_experiment_uses_all_inherit(self):
        from pyramidkernel.config import run_profile

        self.assertEqual(run_profile("quick")["route"], "all_inherit")


if __name__ == "__main__":
    unittest.main()

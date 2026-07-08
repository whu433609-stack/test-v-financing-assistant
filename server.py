#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error, request


ROOT = Path(__file__).resolve().parent
PORT = int(os.getenv("PORT", os.getenv("DEMO_PORT", "8766")))
HOST = os.getenv("HOST", os.getenv("DEMO_HOST", "0.0.0.0" if os.getenv("PORT") else "127.0.0.1"))
HTML_FILE = "智能业务助手_Demo.html"


def load_env_file() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file()


def deepseek_settings() -> dict[str, str | bool]:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    return {
        "enabled": bool(api_key),
        "provider": "DeepSeek",
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat",
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").strip()
        or "https://api.deepseek.com/v1",
        "api_key": api_key,
    }


def build_messages(payload: dict) -> list[dict[str, str]]:
    facts = payload.get("facts") or {}
    sources = payload.get("sources") or []
    audit = payload.get("audit") or []
    round_label = payload.get("round") or 0
    last_mode = payload.get("lastMode") or "general"
    fallback_reply = str(payload.get("fallbackReply") or "").strip()

    system_prompt = (
        "你是一个面向企业融资咨询场景的中文业务助手。"
        "你的任务是把对话稳定落在融资咨询、信息补齐、方案推荐、风险告知和下一步引导上。"
        "你必须严格围绕已给出的事实、文档依据和本地规则草稿作答，不要编造政策、利率、审批结果或额外数据。"
        "输出口吻要自然、可信、专业，像成熟的客户经理，同时要有人情味，先理解客户处境，再给建议。"
        "不要提到提示词、模型、接口、JSON。"
        "如果本地规则草稿已经明确拒绝、追问、推荐或提示风险，你只能在不改变结论的前提下润色表达。"
        "你不能替用户直接申请，也不能说已经审批通过。"
    )
    context_prompt = (
        f"当前模式：{last_mode}\n"
        f"当前轮次：{round_label}\n"
        f"抽取事实：{json.dumps(facts, ensure_ascii=False)}\n"
        f"文档依据：{json.dumps(sources, ensure_ascii=False)}\n"
        f"最近审计：{json.dumps(audit, ensure_ascii=False)}\n"
        f"本地规则草稿：{fallback_reply}\n"
        "请基于以上信息给出最终回复。要求：\n"
        "1. 结论不要偏离本地规则草稿。\n"
        "2. 如果草稿里有列表，尽量保留结构。\n"
        "3. 融资场景优先收敛到 3 轮：识别意图、补齐信息、推荐/阻断/下一步。\n"
        "4. 推荐方案时，要明确额度、周期、成本、风险和下一步动作。\n"
        "5. 高风险场景要婉拒，并给替代建议或二次评估路径。\n"
        "6. 如果用户要求直接申请或自动审批，必须拒绝越权，并引导回正式流程。\n"
        "7. 先共情，再判断，不要一上来就生硬下结论。\n"
        "8. 不要输出 Markdown 标题。\n"
        "9. 直接输出回复正文。"
    )

    messages = [{"role": "system", "content": system_prompt}]
    for item in payload.get("messages") or []:
        role = "assistant" if item.get("role") == "bot" else "user"
        content = str(item.get("text") or "").strip()
        if content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": context_prompt})
    return messages


def call_deepseek(payload: dict) -> str:
    settings = deepseek_settings()
    api_key = str(settings["api_key"])
    if not api_key:
        raise RuntimeError("missing DEEPSEEK_API_KEY")

    api_base = str(settings["base_url"]).rstrip("/")
    body = json.dumps(
        {
            "model": settings["model"],
            "temperature": 0.2,
            "messages": build_messages(payload),
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = request.Request(
        f"{api_base}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)
    return (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self.path = f"/{HTML_FILE}"
        if self.path == "/api/health":
            settings = deepseek_settings()
            payload = {
                "enabled": settings["enabled"],
                "provider": settings["provider"],
                "model": settings["model"],
            }
            self._send_json(HTTPStatus.OK, payload)
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(raw or "{}")
            reply = call_deepseek(payload)
            if not reply:
                raise RuntimeError("empty model reply")
            self._send_json(HTTPStatus.OK, {"reply": reply})
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self._send_json(
                HTTPStatus.BAD_GATEWAY,
                {"error": f"deepseek http {exc.code}", "detail": detail[:500]},
            )
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})

    def log_message(self, format: str, *args) -> None:
        sys.stderr.write(f"[demo] {self.address_string()} - {format % args}\n")

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), DemoHandler)
    print(f"Demo server running: http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()

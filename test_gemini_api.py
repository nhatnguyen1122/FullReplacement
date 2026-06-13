#!/usr/bin/env python3
"""Smoke-test Gemini through Google's OpenAI-compatible API endpoint.

Usage:
    GEMINI_API_KEY="..." python test_gemini_api.py

This intentionally mirrors the endpoint used by the benchmark matrix Gemini
provider, without hardcoding credentials.
"""

from __future__ import annotations

import argparse
import os
import sys

from openai import OpenAI


DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-2.5-flash"


def require_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing GEMINI_API_KEY. Run: GEMINI_API_KEY='your-key' python test_gemini_api.py"
        )
    return api_key


def print_response(label: str, response) -> None:
    message = response.choices[0].message
    content = message.content
    print(f"\n=== {label} ===")
    print(f"model: {response.model}")
    print(f"finish_reason: {response.choices[0].finish_reason}")
    print(f"usage: {response.usage}")
    print("content:")
    print(content if content is not None else "<None>")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--code-test",
        action="store_true",
        help="Also ask Gemini for a code-only Python function, useful for OpenEvolve debugging.",
    )
    args = parser.parse_args()

    client = OpenAI(api_key=require_api_key(), base_url=args.base_url)

    print(f"base_url: {args.base_url}")
    print(f"model: {args.model}")

    sanity = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "Reply with exactly one word: ok"}],
        temperature=0,
        max_tokens=20,
    )
    print_response("sanity", sanity)

    if args.code_test:
        code_prompt = """Return only valid Python code, with no Markdown fences and no explanation.

Define this exact function:

def improve_score(x: float) -> float:
    \"\"\"Return a deterministic value larger than x.\"\"\"
"""
        code = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": code_prompt}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        print_response("code_test", code)

        content = code.choices[0].message.content or ""
        try:
            compile(content, "<gemini_code_test>", "exec")
        except SyntaxError as exc:
            print(f"\ncode_test_compile: FAIL: {exc}", file=sys.stderr)
            return 2
        print("\ncode_test_compile: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

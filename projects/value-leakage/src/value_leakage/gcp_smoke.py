"""Two-call check for visible-reasoning assistant prefill on Vertex GLM-5.2."""

import os
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


PROJECT = os.environ.get("GCP_PROJECT", "<gcp-project>")
MODEL = "zai-org/glm-5.2-maas"
URL = (
    f"https://aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/global/"
    "endpoints/openapi/chat/completions"
)


def access_token() -> str:
    result = subprocess.run(
        ["gcloud", "auth", "print-access-token", "--account=wayman.al@gmail.com"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def request_body(committed: int) -> dict:
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Continue the final assistant message with only its already "
                    "committed integer. Do not recompute, explain, or restart."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Reasoning:\n"
                    f"I have already committed to the exact integer {committed}. "
                    "I must not recompute or choose another value.\n"
                    "Final estimate:"
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 16,
        "stop": ["\n"],
        "chat_template_kwargs": {"enable_thinking": False},
    }


def call(body: dict) -> tuple[int, object]:
    request = Request(
        URL,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {access_token()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        raw = error.read().decode(errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"raw_error": raw}
        return error.code, payload


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("runs") / f"gcp_glm52_content_prefill_{stamp}"
    out.mkdir(parents=True)

    results = []
    for index, committed in enumerate((323, 731), start=1):
        body = request_body(committed)
        status, response = call(body)
        output = (
            response["choices"][0]["message"]["content"]
            if status == 200
            else None
        )
        record = {
            "provider": "gcp",
            "project": PROJECT,
            "location": "global",
            "url": URL,
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "expected_continuation": str(committed),
            "request": body,
            "http_status": status,
            "response": response,
        }
        (out / f"call_{index}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False)
        )
        results.append({"expected": str(committed), "status": status, "output": output})

    summary = {
        "results": results,
        "passed": all(
            r["status"] == 200 and r["output"].strip() == r["expected"]
            for r in results
        ),
        "artifacts": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Task 4 demo: auth status codes + pytest with coverage, printed in the expected format."""
import os
import re
import subprocess
import sys

os.environ["API_KEY"] = "demo-key"
os.environ["USE_MOCK_LLM"] = "true"

from fastapi.testclient import TestClient   # noqa: E402

from app.main import app                      # noqa: E402


def main():
    client = TestClient(app)
    no_key = client.post("/sessions").status_code
    wrong_key = client.post("/sessions", headers={"x-api-key": "wrong"}).status_code

    proc = subprocess.run([sys.executable, "-m", "pytest", "--cov=app", "--cov-report=term"],
                          capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    passed = re.search(r"(\d+) passed", out)
    cov = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", out)
    print(f"{passed.group(1) if passed else '?'} passed, "
          f"coverage {cov.group(1) + '%' if cov else 'n/a'} | No key -> {no_key} | Wrong key -> {wrong_key}")
    if proc.returncode != 0:
        print(out)


if __name__ == "__main__":
    main()

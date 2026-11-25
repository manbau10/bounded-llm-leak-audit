"""Minimal LLM clients. Keys are read from environment variables (GEMINI_API_KEY, POE_API_KEY) and never stored."""
import json, os, time, urllib.request, urllib.error

def gemini_key(): return os.environ["GEMINI_API_KEY"]
def poe_key(): return os.environ["POE_API_KEY"]

def _post(url, headers, body, timeout=180, retries=4):
    for a in range(retries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
            return json.load(urllib.request.urlopen(req, timeout=timeout))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and a < retries - 1: time.sleep(5 * (a + 1)); continue
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:300]}")

def poe(prompt, model, temperature=0.0, max_tokens=20000):
    r = _post("https://api.poe.com/v1/chat/completions", {"Authorization": "Bearer " + poe_key()},
              {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature, "max_tokens": max_tokens})
    return r["choices"][0]["message"]["content"]

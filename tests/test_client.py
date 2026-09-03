import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from autokv.client import VllmClient


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def _json(self, payload):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path == "/health":
            self._json({"status": "ok"})
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        if self.path == "/tokenize":
            self._json({"count": len(body["prompt"].split()), "tokens": [1, 2]})
        elif self.path == "/v1/completions":
            self._json({"choices": [{"text": "answer"}]})
        elif self.path == "/v1/chat/completions":
            if body != {
                "model": "model",
                "messages": [{"role": "user", "content": "question"}],
                "max_tokens": 12,
                "temperature": 0,
                "top_p": 1,
                "seed": 42,
                "stream": False,
            }:
                self.send_error(400)
                return
            self._json(
                {
                    "choices": [{"message": {"content": "chat answer"}}],
                    "usage": {"prompt_tokens": 7},
                }
            )
        else:
            self.send_error(404)


class ClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_completion_payload_is_deterministic(self):
        payload = VllmClient.completion_payload("model", "prompt", 24)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["max_tokens"], 24)

    def test_health_tokenize_and_completion_use_real_http(self):
        client = VllmClient(self.base_url, "model", timeout=2)
        self.assertTrue(client.health())
        self.assertEqual(client.tokenize("one two three"), 3)
        response = client.complete("prompt", 24)
        self.assertEqual(response["choices"][0]["text"], "answer")
        chat = client.chat_complete("question", 12)
        self.assertEqual(chat["choices"][0]["message"]["content"], "chat answer")
        self.assertEqual(chat["usage"]["prompt_tokens"], 7)


if __name__ == "__main__":
    unittest.main()

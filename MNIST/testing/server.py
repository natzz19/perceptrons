"""Tiny bridge: browser -> C binary.

GET  /         the drawing page (index.html, must sit next to this file)
POST /predict  {"pixels": [784 floats, 0..1]}  ->  {"probs": [10 floats]}

MNIST_processDrawing takes in 784 whitespace-separated floats
from stdin and prints 10 whitespace-separated probabilities to stdout.

"""
import json, subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer

BINARY_CMD = ["../../builds/MNIST_processDrawing", "../out/mnist_mlp_128_softmax.CE.weights"]   # <-- change me
SCALE = 1.0                                                                                     # 255.0 if your binary expects 0..255

class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_GET(self):
        # Serve the page itself, so http://localhost:8000 just works.
        try:
            data = open("index.html", "rb").read()
        except FileNotFoundError:
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)

    def do_POST(self):
        if self.path != "/predict":
            self.send_response(404); self._cors(); self.end_headers(); return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        pixels = body["pixels"]
        assert len(pixels) == 784
        stdin = " ".join(f"{p * SCALE:.6f}" for p in pixels)
        out = subprocess.run(BINARY_CMD, input=stdin, capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            print(out.stderr)
            self.send_response(500); self._cors(); self.end_headers(); return
        probs = [float(v) for v in out.stdout.split()][:10]
        data = json.dumps({"probs": probs}).encode()
        self.send_response(200); self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)

    def log_message(self, *a): pass

if __name__ == "__main__":
    print("listening on http://localhost:8000")
    HTTPServer(("localhost", 8000), Handler).serve_forever()

#!/usr/bin/env python3
"""Central HTTP distributor for Flux worker servers."""

import argparse
import random
import socket
import subprocess
import sys
import threading
import time

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

parser = argparse.ArgumentParser(description="Flux Central Distribution Server")
parser.add_argument("worker_urls", nargs="?", default="", help="Comma-separated worker base URLs")
parser.add_argument("--port", type=int, default=8090, help="Central server port")
args = parser.parse_args()

worker_servers = [url.strip() for url in args.worker_urls.split(",") if url.strip()]
current_worker = 0
worker_lock = threading.Lock()

central_jobs = {}
central_jobs_lock = threading.Lock()

app = Flask(__name__)
CORS(app)


def get_network_ip():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip_address = sock.getsockname()[0]
        sock.close()
        return ip_address
    except OSError:
        try:
            return subprocess.check_output(["hostname", "-I"], text=True).strip().split()[0]
        except (subprocess.CalledProcessError, IndexError):
            return "0.0.0.0"


def get_next_worker():
    global current_worker
    if not worker_servers:
        return None
    with worker_lock:
        worker = worker_servers[current_worker % len(worker_servers)]
        current_worker += 1
        return worker


def check_worker_health(worker_url):
    try:
        response = requests.get(f"{worker_url}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


@app.route("/", methods=["GET"])
def root():
    network_ip = get_network_ip()
    port = args.port
    return jsonify(
        {
            "service": "Flux Image Generation API (Central Distributor)",
            "version": "1.0.0",
            "status": "running",
            "worker_servers": len(worker_servers),
            "endpoints": {
                "health": f"http://{network_ip}:{port}/health",
                "generate": f"http://{network_ip}:{port}/generate",
            },
        }
    )


@app.route("/health", methods=["GET"])
def health_check():
    healthy_workers = sum(1 for worker in worker_servers if check_worker_health(worker))
    return jsonify(
        {
            "status": "healthy" if healthy_workers > 0 else "degraded",
            "total_workers": len(worker_servers),
            "healthy_workers": healthy_workers,
            "gpu_available": healthy_workers > 0,
        }
    )


@app.route("/metrics", methods=["GET"])
def metrics():
    total_metrics = {"workers": len(worker_servers), "healthy_workers": 0, "worker_metrics": []}
    for worker_url in worker_servers:
        try:
            response = requests.get(f"{worker_url}/metrics", timeout=5)
            if response.status_code == 200:
                total_metrics["worker_metrics"].append(
                    {"url": worker_url, "metrics": response.json(), "status": "healthy"}
                )
                total_metrics["healthy_workers"] += 1
            else:
                total_metrics["worker_metrics"].append({"url": worker_url, "status": "unhealthy"})
        except requests.RequestException:
            total_metrics["worker_metrics"].append({"url": worker_url, "status": "unreachable"})
    return jsonify(total_metrics)


@app.route("/openapi.json", methods=["GET"])
@app.route("/api/v1/models", methods=["GET"])
@app.route("/docs", methods=["GET"])
def forward_get():
    worker = get_next_worker()
    if not worker:
        return jsonify({"error": "No healthy workers available"}), 503
    path = request.path
    try:
        response = requests.get(f"{worker}{path}", timeout=10)
        if response.status_code != 200:
            return jsonify({"error": f"Worker returned {response.status_code}"}), 500
        if path == "/docs":
            return response.text, 200, {"Content-Type": "text/html"}
        return response.json()
    except requests.RequestException:
        return jsonify({"error": "Worker unreachable"}), 503


@app.route("/generate", methods=["POST"])
def generate_image():
    try:
        data = request.get_json()
        job_id = f"central_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"

        with central_jobs_lock:
            central_jobs[job_id] = {
                "status": "processing",
                "request_data": data,
                "created_at": time.time(),
            }

        def process_async():
            attempts = 0
            max_attempts = len(worker_servers) if worker_servers else 1

            while attempts < max_attempts:
                worker = get_next_worker()
                if not worker:
                    with central_jobs_lock:
                        central_jobs[job_id]["status"] = "failed"
                        central_jobs[job_id]["error"] = "No workers available"
                    return

                try:
                    response = requests.post(f"{worker}/generate", json=data, timeout=10)
                    if response.status_code != 202:
                        attempts += 1
                        continue

                    worker_job_id = response.json().get("job_id")
                    while True:
                        time.sleep(2)
                        status_response = requests.get(
                            f"{worker}/job/{worker_job_id}",
                            timeout=10,
                        )
                        if status_response.status_code == 200:
                            with central_jobs_lock:
                                central_jobs[job_id]["status"] = "completed"
                                central_jobs[job_id]["file_content"] = status_response.content
                                central_jobs[job_id]["completed_at"] = time.time()
                            return
                        if status_response.status_code == 500:
                            with central_jobs_lock:
                                central_jobs[job_id]["status"] = "failed"
                                central_jobs[job_id]["error"] = status_response.json().get(
                                    "error", "Worker generation failed"
                                )
                            return
                except requests.RequestException:
                    attempts += 1

            with central_jobs_lock:
                central_jobs[job_id]["status"] = "failed"
                central_jobs[job_id]["error"] = "All workers failed or unavailable"

        threading.Thread(target=process_async, daemon=True).start()
        return jsonify(
            {
                "status": "accepted",
                "job_id": job_id,
                "message": "Request received and processing started",
            }
        ), 202
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/job/<job_id>", methods=["GET"])
def get_job_status(job_id):
    with central_jobs_lock:
        job = central_jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] == "completed":
        return app.response_class(
            job["file_content"],
            mimetype="image/png",
            headers={"Content-Disposition": "attachment; filename=image.png"},
        )
    if job["status"] == "failed":
        return jsonify({"status": "failed", "error": job.get("error", "Unknown error")}), 500
    return jsonify(
        {"status": job["status"], "job_id": job_id, "message": "Job is still processing"}
    ), 202


if __name__ == "__main__":
    if not worker_servers and len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        worker_servers[:] = [url.strip() for url in sys.argv[1].split(",") if url.strip()]

    network_ip = get_network_ip()
    port = args.port
    print("=" * 60)
    print("Flux Central Distribution Server")
    print(f"Local:   http://localhost:{port}")
    print(f"Network: http://{network_ip}:{port}")
    print(f"Workers: {worker_servers}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)

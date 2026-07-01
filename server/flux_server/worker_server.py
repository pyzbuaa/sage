#!/usr/bin/env python3
"""Flux image generation worker (one GPU per process)."""

import argparse
import io
import os
import random
import socket
import subprocess
import threading
import time

import psutil
import torch
from diffusers import FluxPipeline
from flask import Flask, jsonify, redirect, request, send_file
from flask_cors import CORS

parser = argparse.ArgumentParser(description="Flux Image Generation Server")
parser.add_argument("--port", type=int, default=8091, help="Port to run the server on")
parser.add_argument("--gpu", type=int, default=None, help="GPU ID (for logging; use CUDA_VISIBLE_DEVICES)")
args = parser.parse_args()

if args.gpu is not None:
    print(f"Running on GPU {args.gpu} (via CUDA_VISIBLE_DEVICES)")
else:
    print("Using default GPU configuration")

app = Flask(__name__)
CORS(app)

start_time = time.time()
request_count = 0
error_count = 0
total_generation_time = 0.0

print("Loading Flux pipeline...", flush=True)
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Krea-dev",
    torch_dtype=torch.bfloat16,
)
pipeline.enable_model_cpu_offload()
print("Pipeline loaded successfully!", flush=True)


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


def get_openapi_spec():
    network_ip = get_network_ip()
    port = args.port
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Flux Image Generation API",
            "version": "1.0.0",
            "description": "API for generating images using Flux model",
        },
        "servers": [{"url": f"http://{network_ip}:{port}"}],
        "paths": {
            "/health": {"get": {"summary": "Health check"}},
            "/generate": {"post": {"summary": "Generate image from text"}},
        },
    }


@app.route("/", methods=["GET"])
def root():
    network_ip = get_network_ip()
    port = args.port
    return jsonify(
        {
            "service": "Flux Image Generation API",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "health": f"http://{network_ip}:{port}/health",
                "generate": f"http://{network_ip}:{port}/generate",
            },
        }
    )


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "gpu_available": torch.cuda.is_available()})


@app.route("/metrics", methods=["GET"])
def metrics():
    global request_count, error_count, total_generation_time, start_time

    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_count": torch.cuda.device_count(),
            "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "gpu_name": torch.cuda.get_device_name(0),
        }

    memory = psutil.virtual_memory()
    uptime = time.time() - start_time
    avg_generation_time = total_generation_time / max(1, request_count)

    return jsonify(
        {
            "gpu": gpu_info,
            "application": {
                "uptime_seconds": uptime,
                "total_requests": request_count,
                "total_errors": error_count,
                "average_generation_time_seconds": avg_generation_time,
            },
            "system": {
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / 1024**3,
                "memory_total_gb": memory.total / 1024**3,
            },
            "timestamp": time.time(),
        }
    )


@app.route("/openapi.json", methods=["GET"])
def openapi_spec():
    response = jsonify(get_openapi_spec())
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.route("/api/v1/models", methods=["GET"])
def list_models():
    return jsonify(
        {
            "object": "list",
            "data": [
                {
                    "id": "black-forest-labs/FLUX.1-Krea-dev",
                    "object": "model",
                    "owned_by": "black-forest-labs",
                }
            ],
        }
    )


@app.route("/docs", methods=["GET"])
def api_docs():
    network_ip = get_network_ip()
    port = args.port
    return (
        f"<h1>Flux API</h1><p>POST http://{network_ip}:{port}/generate</p>",
        200,
        {"Content-Type": "text/html"},
    )


@app.route("/swagger", methods=["GET"])
def swagger_redirect():
    return redirect("/docs")


worker_jobs = {}
worker_jobs_lock = threading.Lock()


@app.route("/generate", methods=["POST"])
def generate_image():
    global request_count, error_count, total_generation_time

    request_count += 1

    try:
        data = request.get_json() or {}
        prompt = data.get("prompt", "A simple image")
        height = data.get("height", 1024)
        width = data.get("width", 1024)
        guidance_scale = data.get("guidance_scale", 4.5)
        seed = data.get("seed", random.randint(1, 1_000_000))
        job_id = f"{int(time.time() * 1000)}_{seed}"

        with worker_jobs_lock:
            worker_jobs[job_id] = {
                "status": "processing",
                "prompt": prompt,
                "seed": seed,
                "created_at": time.time(),
            }

        print(f"Job {job_id}: acknowledged: {prompt}", flush=True)

        def generate_async():
            global total_generation_time, error_count
            generation_start = time.time()
            try:
                generator = torch.Generator().manual_seed(seed)
                image = pipeline(
                    prompt,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images[0]

                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="PNG")
                file_content = img_byte_arr.getvalue()
                generation_end = time.time()
                total_generation_time += generation_end - generation_start

                with worker_jobs_lock:
                    worker_jobs[job_id] = {
                        "status": "completed",
                        "seed": seed,
                        "created_at": worker_jobs[job_id]["created_at"],
                        "completed_at": time.time(),
                        "file_content": file_content,
                        "generation_time": generation_end - generation_start,
                    }
                print(f"Job {job_id}: completed", flush=True)
            except Exception as exc:
                error_count += 1
                print(f"Job {job_id}: failed: {exc}", flush=True)
                with worker_jobs_lock:
                    worker_jobs[job_id] = {
                        "status": "failed",
                        "seed": seed,
                        "created_at": worker_jobs[job_id]["created_at"],
                        "error": str(exc),
                    }

        threading.Thread(target=generate_async, daemon=True).start()
        return jsonify(
            {
                "status": "accepted",
                "job_id": job_id,
                "message": "Request received and processing started",
            }
        ), 202
    except Exception as exc:
        error_count += 1
        return jsonify({"error": str(exc)}), 500


@app.route("/job/<job_id>", methods=["GET"])
def get_job_status(job_id):
    with worker_jobs_lock:
        job = worker_jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] == "completed":
        return send_file(
            io.BytesIO(job["file_content"]),
            as_attachment=True,
            download_name=f"generated_image_{job['seed']}.png",
            mimetype="image/png",
        )
    if job["status"] == "failed":
        return jsonify({"status": "failed", "error": job.get("error", "Unknown error")}), 500
    return jsonify(
        {"status": job["status"], "job_id": job_id, "message": "Job is still processing"}
    ), 202


if __name__ == "__main__":
    port = args.port
    network_ip = get_network_ip()
    print("=" * 60)
    print("Flux Image Generation Worker")
    print(f"Local:   http://localhost:{port}")
    print(f"Network: http://{network_ip}:{port}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)

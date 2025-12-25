#!/usr/bin/env python3
"""
Benchmark script for Streamware Jetson

Testuje wydajność poszczególnych komponentów:
- STT (Faster-Whisper)
- Vision (YOLOv8)
- LLM (Ollama)
- TTS (Piper)
"""

import argparse
import asyncio
import time
import statistics
from pathlib import Path
import sys

import numpy as np


def print_header(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_results(name: str, times: list):
    if not times:
        print(f"  {name}: No data")
        return
    
    avg = statistics.mean(times) * 1000
    std = statistics.stdev(times) * 1000 if len(times) > 1 else 0
    min_t = min(times) * 1000
    max_t = max(times) * 1000
    
    print(f"  {name}:")
    print(f"    Mean:   {avg:.1f} ms")
    print(f"    Std:    {std:.1f} ms")
    print(f"    Min:    {min_t:.1f} ms")
    print(f"    Max:    {max_t:.1f} ms")


async def benchmark_stt(iterations: int = 10):
    """Benchmark STT (Faster-Whisper)."""
    print_header("STT Benchmark (Faster-Whisper)")
    
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("  ❌ faster-whisper not installed")
        return
    
    print("  Loading model...")
    model = WhisperModel("small", device="cuda", compute_type="float16")
    
    sample_rate = 16000
    duration = 3
    audio = np.random.randn(sample_rate * duration).astype(np.float32) * 0.01
    
    print("  Warmup...")
    for _ in range(3):
        list(model.transcribe(audio, language="pl"))
    
    print(f"  Running {iterations} iterations...")
    times = []
    
    for i in range(iterations):
        start = time.perf_counter()
        list(model.transcribe(audio, language="pl", beam_size=5))
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    print_results("Whisper small (3s audio)", times)
    print(f"    RTF:    {statistics.mean(times) / duration:.2f}x realtime")


async def benchmark_vision(iterations: int = 50):
    """Benchmark Vision (YOLOv8)."""
    print_header("Vision Benchmark (YOLOv8)")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ❌ ultralytics not installed")
        return
    
    engine_path = Path("models/yolo/yolov8n.engine")
    
    if engine_path.exists():
        print(f"  Using TensorRT: {engine_path}")
        model = YOLO(str(engine_path))
    else:
        print("  Using PyTorch (TensorRT recommended)")
        model = YOLO("yolov8n.pt")
    
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("  Warmup...")
    for _ in range(10):
        model(img, verbose=False)
    
    print(f"  Running {iterations} iterations...")
    times = []
    
    for i in range(iterations):
        start = time.perf_counter()
        model(img, verbose=False)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    print_results("YOLOv8n (640x480)", times)
    fps = 1.0 / statistics.mean(times)
    print(f"    FPS:    {fps:.1f}")


async def benchmark_llm(iterations: int = 5):
    """Benchmark LLM (Ollama)."""
    print_header("LLM Benchmark (Ollama)")
    
    try:
        import httpx
    except ImportError:
        print("  ❌ httpx not installed")
        return
    
    async with httpx.AsyncClient(base_url="http://localhost:11434", timeout=60.0) as client:
        try:
            resp = await client.get("/api/tags")
            if resp.status_code != 200:
                print("  ❌ Ollama not running")
                return
        except:
            print("  ❌ Cannot connect to Ollama")
            return
        
        model = "phi3:mini"
        prompt = "Odpowiedz jednym zdaniem: co to jest sztuczna inteligencja?"
        
        print(f"  Model: {model}")
        
        print("  Warmup...")
        await client.post("/api/chat", json={
            "model": model,
            "messages": [{"role": "user", "content": "Cześć"}],
            "stream": False
        })
        
        print(f"  Running {iterations} iterations...")
        times = []
        tokens_per_sec = []
        
        for i in range(iterations):
            start = time.perf_counter()
            
            resp = await client.post("/api/chat", json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"num_predict": 64}
            })
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
            data = resp.json()
            eval_count = data.get("eval_count", 0)
            eval_duration = data.get("eval_duration", 1) / 1e9
            if eval_duration > 0:
                tokens_per_sec.append(eval_count / eval_duration)
        
        print_results("phi3:mini (64 tokens)", times)
        if tokens_per_sec:
            print(f"    Tokens/s: {statistics.mean(tokens_per_sec):.1f}")


async def benchmark_tts(iterations: int = 10):
    """Benchmark TTS (Piper)."""
    print_header("TTS Benchmark (Piper)")
    
    model_path = Path("models/piper/pl_PL-gosia-medium.onnx")
    
    if not model_path.exists():
        print(f"  ❌ Model not found: {model_path}")
        print("  Run: ./scripts/download_piper_pl.sh")
        return
    
    try:
        from piper import PiperVoice
    except ImportError:
        print("  ❌ piper-tts not installed")
        return
    
    print("  Loading model...")
    voice = PiperVoice.load(
        str(model_path),
        str(model_path.with_suffix(".onnx.json"))
    )
    
    text = "Dzień dobry, jestem asystentem wizyjno-głosowym."
    
    print("  Warmup...")
    for _ in range(3):
        list(voice.synthesize_stream_raw(text))
    
    print(f"  Running {iterations} iterations...")
    times = []
    
    for i in range(iterations):
        start = time.perf_counter()
        audio = list(voice.synthesize_stream_raw(text))
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    print_results("Piper gosia-medium", times)


async def benchmark_e2e():
    """End-to-end latency estimation."""
    print_header("End-to-End Latency Estimate")
    
    print("""
  Based on individual benchmarks:
  
  Component         | Typical Latency
  ------------------|----------------
  Audio capture     | ~30 ms
  VAD processing    | ~10 ms
  STT (small, 3s)   | ~200-400 ms
  Vision (YOLOv8n)  | ~20-50 ms
  LLM (phi3:mini)   | ~300-600 ms
  TTS (Piper)       | ~50-100 ms
  Audio playback    | ~30 ms
  ------------------|----------------
  TOTAL             | ~640-1220 ms
  
  Target: < 1000 ms for responsive interaction
    """)


async def main():
    parser = argparse.ArgumentParser(description="Streamware Jetson Benchmark")
    parser.add_argument("--all", "-a", action="store_true", help="Run all benchmarks")
    parser.add_argument("--stt", action="store_true", help="Benchmark STT")
    parser.add_argument("--vision", action="store_true", help="Benchmark Vision")
    parser.add_argument("--llm", action="store_true", help="Benchmark LLM")
    parser.add_argument("--tts", action="store_true", help="Benchmark TTS")
    parser.add_argument("--iterations", "-n", type=int, default=10, help="Iterations")
    
    args = parser.parse_args()
    
    print()
    print("=" * 60)
    print("  Streamware Jetson - Performance Benchmark")
    print("=" * 60)
    
    if args.all or (not args.stt and not args.vision and not args.llm and not args.tts):
        await benchmark_stt(args.iterations)
        await benchmark_vision(args.iterations * 5)
        await benchmark_llm(args.iterations // 2 or 3)
        await benchmark_tts(args.iterations)
        await benchmark_e2e()
    else:
        if args.stt:
            await benchmark_stt(args.iterations)
        if args.vision:
            await benchmark_vision(args.iterations * 5)
        if args.llm:
            await benchmark_llm(args.iterations // 2 or 3)
        if args.tts:
            await benchmark_tts(args.iterations)
    
    print()
    print("Benchmark complete!")


if __name__ == "__main__":
    asyncio.run(main())

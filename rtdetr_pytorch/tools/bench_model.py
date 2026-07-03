import argparse, io, time, torch
from fvcore.nn import FlopCountAnalysis
from src.core.yaml_config import YAMLConfig

def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def model_size_mb(model):
    buf = io.BytesIO(); torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / (1024*1024)

@torch.no_grad()
def bench_latency(model, h=640, w=640, device="cpu", warmup=10, runs=50, threads=4):
    torch.set_num_threads(threads)
    model.eval().to(device)
    x = torch.zeros(1, 3, h, w, device=device)
    for _ in range(warmup): _ = model(x)
    t0 = time.perf_counter()
    for _ in range(runs): _ = model(x)
    dt = (time.perf_counter() - t0) * 1000.0 / runs
    return dt, 1000.0 / dt  # ms, FPS

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--size", type=int, default=640)
    args = ap.parse_args()

    cfg = YAMLConfig(args.config)
    model = cfg.model  # 已按你的框架 create()
    params_m = count_params(model)
    size_mb = model_size_mb(model)

    x = torch.zeros(1,3,args.size,args.size)
    flops = FlopCountAnalysis(model, x).total() / 1e9  # GFLOPs（一次前向）
    lat_ms, fps = bench_latency(model, args.size, args.size, args.device, threads=args.threads)

    print(f"Params/M = {params_m:.2f}")
    print(f"Size (MB) = {size_mb:.2f}")
    print(f"GFLOPs/G = {flops:.1f}  (input={args.size}x{args.size}, bs=1)")
    print(f"Latency = {lat_ms:.2f} ms  |  FPS = {fps:.0f}")

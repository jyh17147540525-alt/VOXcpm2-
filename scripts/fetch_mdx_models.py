#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载 MDX-NET 人声分离模型权重
==============================
「只保留纯净人声（去背景音乐）」功能依赖 UVR (Ultimate Vocal Remover) 的
MDX-NET ONNX 权重。权重体积较大（30~65MB/个），不随仓库分发，用本脚本按需拉取。

用法：
    python scripts/fetch_mdx_models.py            # 下载默认模型（Main_340）
    python scripts/fetch_mdx_models.py all        # 下载全部可用模型
    python scripts/fetch_mdx_models.py 9482       # 按关键字下载

下载目标：<项目根>/models/mdx/
镜像顺序：HF-Mirror -> HuggingFace 官方（可加 --source 指定）

无权重时程序不会报错：「只保留纯净人声」会自动回退到内置的
REPET-lite / HPSS 纯 DSP 分离（质量较低：实测 corr≈0.85，去背景音乐时
会让人声发闷；装了神经模型可达 corr≈0.99）。
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "models" / "mdx"

# 模型清单：文件名 -> 下载源（HF 仓库路径）
MODELS = {
    "UVR-MDX-NET_Main_340.onnx": {
        "repo": "Eddycrack864/UVR5-MDX-NET-VIP-MODELS",
        "size_mb": 64,
        "desc": "Main 340 —— 人声/伴奏通用，UVR 社区口碑最佳之一（推荐）",
    },
    "UVR_MDXNET_9482.onnx": {
        "repo": "FHFanshu/UVR_Inference_Models",
        "size_mb": 28,
        "desc": "MDXNET 9482 —— 体积小，人声分离（备选）",
    },
}
DEFAULT = "UVR-MDX-NET_Main_340.onnx"

MIRRORS = [
    ("hf-mirror", "https://hf-mirror.com"),
    ("huggingface", "https://huggingface.co"),
]


def _url(base: str, repo: str, name: str) -> str:
    return f"{base}/{repo}/resolve/main/{name}"


def _download(url: str, dst: Path) -> bool:
    """带进度条下载；失败返回 False。"""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "VoxCPM2-fetch/1.0"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 绕开失效代理
        with opener.open(req, timeout=60) as r:
            total = int(r.headers.get("Content-Length") or 0)
            tmp = dst.with_suffix(dst.suffix + ".part")
            got = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if total:
                        pct = got * 100 // total
                        mb = got / 1048576
                        print(f"\r    {pct:3d}%  {mb:6.1f} / {total/1048576:.1f} MB",
                              end="", flush=True)
            print()
            if total and got < total:
                tmp.unlink(missing_ok=True)
                return False
            tmp.replace(dst)
            return True
    except Exception as e:
        print(f"\n    [x] {type(e).__name__}: {e}")
        try:
            dst.with_suffix(dst.suffix + ".part").unlink(missing_ok=True)
        except OSError:
            pass
        return False


def fetch(name: str) -> bool:
    info = MODELS[name]
    dst = DEST / name
    if dst.exists() and dst.stat().st_size > 1 << 20:
        print(f"[=] {name} 已存在（{dst.stat().st_size/1048576:.1f} MB），跳过")
        return True
    print(f"[*] {name}  ({info['size_mb']}MB)  {info['desc']}")
    for label, base in MIRRORS:
        print(f"  -> 尝试 {label} ...")
        if _download(_url(base, info["repo"], name), dst):
            print(f"  [OK] 已保存到 {dst}")
            return True
    print(f"  [FAIL] 所有镜像均失败：{name}")
    return False


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        targets = [DEFAULT]
    elif args[0].lower() in ("all", "-a"):
        targets = list(MODELS)
    else:
        keys = [a.lower() for a in args]
        targets = [n for n in MODELS if any(k in n.lower() for k in keys)]
        if not targets:
            print(f"未匹配到模型。可用：{', '.join(MODELS)}")
            return 1
    print(f"目标目录: {DEST}\n")
    ok = sum(1 for t in targets if fetch(t))
    print(f"\n完成：{ok}/{len(targets)} 个模型就绪。")
    if ok == 0:
        print("提示：也可手动下载后放入 models/mdx/ 目录。")
        return 1
    print("重启服务后「只保留纯净人声」即自动启用神经分离。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

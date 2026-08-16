"""
命令行入口：
  python -m voice_clone.cli prepare --input ref.wav [--no-denoise] [--remove-bg] [--target-dur 25]
  python -m voice_clone.cli demo   --input ref.wav --text "..."   # 需已加载模型，一般不离线用
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser(description="VoxCPM2 长参考音频克隆增强")
    sub = p.add_subparsers(dest="cmd")

    pp = sub.add_parser("prepare", help="准备融合参考音频")
    pp.add_argument("--input", required=True)
    pp.add_argument("--no-denoise", action="store_true")
    pp.add_argument("--remove-bg", action="store_true")
    pp.add_argument("--target-dur", type=float, default=25.0)

    dt = sub.add_parser("demo", help="端到端示例(需模型)")
    dt.add_argument("--input", required=True)
    dt.add_argument("--text", default="这是一段用于验证的测试文本。")
    dt.add_argument("--no-denoise", action="store_true")
    dt.add_argument("--remove-bg", action="store_true")

    args = p.parse_args()
    if args.cmd == "prepare":
        from voice_clone import prepare_reference
        path, rep = prepare_reference(args.input, denoise=not args.no_denoise,
                                      remove_bg=args.remove_bg, target_dur=args.target_dur)
        print("融合参考路径:", path)
        print("报告:", rep)
    elif args.cmd == "demo":
        from voxcpm import VoxCPM
        from voice_clone import run_demo
        m = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, device="auto")
        res = run_demo(args.input, args.text, model=m,
                      denoise=not args.no_denoise, remove_bg=args.remove_bg)
        print(res)
    else:
        p.print_help()


if __name__ == "__main__":
    main()

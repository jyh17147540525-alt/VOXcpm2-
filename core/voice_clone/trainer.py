"""
LoRA 训练引擎 (Trainer)
=======================
基于 VoxCPM 官方训练基建（voxcpm.training 的 dataset / packer / dataloader）+
模型原生 LoRA（LoRAConfig / _apply_lora / get_lora_state_dict），实现单卡 LoRA
微调循环。设计目标：在 16GB 消费级显卡（4070 Ti SUPER）上，用少量「语音+台词」
样本对音色/风格做持续增量学习，不破坏原模型。

流程：
  1. 从 training_store 导出 JSONL manifest（text + audio）
  2. 卸载推理模型，释放显存
  3. 以 LoRA 配置重新加载模型（optimize=False，不需要 triton/torch.compile）
  4. HF dataset（自动重采样 + 分词）-> dataloader -> BatchProcessor 打包
  5. 训练循环：forward(loss/diff + loss/stop) -> backward -> AdamW(仅 LoRA 参数)
  6. 保存 lora_weights.safetensors + meta.json，卸载训练模型

状态与控制：
  start_training(cfg)  后台线程启动
  get_status()         查询进度/损失
  request_stop()       优雅停止（保存当前权重）
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path

import torch

import voice_clone.training_store as tstore

# ------------------------------------------------------------------ 全局状态
_state_lock = threading.Lock()
_state: dict = {"running": False, "status": "idle", "message": "", "error": None}
_stop_flag = threading.Event()
_thread: threading.Thread | None = None

DEFAULTS = {
    "lora_name": "",
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "enable_lm": True,
    "enable_dit": True,
    "enable_proj": True,
    "lr": 1e-4,
    "epochs": 3,
    "batch_size": 1,
    "grad_accum": 4,
    "max_samples": 200,
}


def get_status() -> dict:
    with _state_lock:
        return dict(_state)


def is_running() -> bool:
    with _state_lock:
        return bool(_state.get("running"))


def request_stop():
    _stop_flag.set()
    with _state_lock:
        _state["message"] = "收到停止请求，将在当前步完成后保存并退出…"


def _update(**kw):
    with _state_lock:
        _state.update(kw)


# ------------------------------------------------------------------ 核心
def start_training(cfg: dict | None = None, base_dir: Path | None = None) -> dict:
    """在后台线程启动一次 LoRA 训练。返回当前状态快照。

    注意：调用方（server）应在调用前先卸载推理模型（unload_model）释放显存；
    本函数只做兜底清理。"""
    global _thread
    if is_running():
        raise RuntimeError("已有训练任务在运行中")
    if tstore.TRAIN_DIR is None:
        tstore.init(base_dir)

    full = dict(DEFAULTS)
    full.update({k: v for k, v in (cfg or {}).items() if v is not None})
    samples = tstore.list_samples()
    if len(samples) < 2:
        raise ValueError("训练样本不足：至少需要 2 条「语音+台词」样本")

    _stop_flag.clear()
    _update(running=True, status="preparing", progress=0, message="准备训练…",
            error=None, loss_history=[], config={k: full[k] for k in DEFAULTS})

    _thread = threading.Thread(target=_train_worker, args=(full,), daemon=True)
    _thread.start()
    return get_status()


def _train_worker(cfg: dict):
    """训练主流程（后台线程）。"""
    from voxcpm import VoxCPM
    from voxcpm.model.voxcpm import LoRAConfig

    MODEL_PATH = Path(__file__).resolve().parent.parent  # F:\VoxCPM2
    t0 = time.time()
    try:
        # ---------- 1. 导出 manifest ----------
        manifest = tstore.export_manifest()
        n_lines = len([l for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()])
        if n_lines < 2:
            raise ValueError("有效样本不足 2 条，无法训练")
        _update(message=f"已导出 {n_lines} 条训练样本", progress=2)

        # ---------- 2. 卸载推理模型由调用方完成；这里兜底清理显存 ----------
        _update(message="清理显存…")
        try:
            import gc as _g
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _g.collect()
        except Exception:
            pass

        # ---------- 3. 以 LoRA 配置加载模型 ----------
        _update(message="加载模型（LoRA 模式）…", progress=5)
        lora_cfg = LoRAConfig(
            enable_lm=bool(cfg["enable_lm"]),
            enable_dit=bool(cfg["enable_dit"]),
            enable_proj=bool(cfg["enable_proj"]),
            r=int(cfg["lora_r"]),
            alpha=int(cfg["lora_alpha"]),
            dropout=float(cfg["lora_dropout"]),
        )
        core = VoxCPM.from_pretrained(
            str(MODEL_PATH), load_denoiser=False, optimize=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
            lora_config=lora_cfg,
        )
        tts = core.tts_model
        sr = int(getattr(tts, "sample_rate", 48000))
        tts.train()
        # audio_vae 只做编码，保持 eval + 冻结
        if hasattr(tts, "audio_vae") and tts.audio_vae is not None:
            tts.audio_vae.eval()
            for p in tts.audio_vae.parameters():
                p.requires_grad_(False)

        # 冻结非 LoRA 参数
        lora_params = []
        for name, p in tts.named_parameters():
            if "lora_" in name:
                p.requires_grad_(True)
                lora_params.append(p)
            else:
                p.requires_grad_(False)
        n_lora = sum(p.numel() for p in lora_params)
        _update(message=f"LoRA 参数量: {n_lora/1e6:.2f}M / 总参数冻结", progress=8,
                lora_params_m=n_lora / 1e6)

        # ---------- 4. 数据集 ----------
        from voxcpm.training.data import (
            load_audio_text_datasets, BatchProcessor, build_dataloader,
        )
        from voxcpm.training.accelerator import Accelerator

        ds, _ = load_audio_text_datasets(train_manifest=str(manifest), sample_rate=sr)
        tokenizer = tts.text_tokenizer

        def _tok(batch):
            return {"text_ids": [list(tokenizer(t)) for t in batch["text"]]}

        keep = [c for c in ds.column_names if c in ("audio", "dataset_id", "is_prompt", "ref_audio")]
        ds = ds.map(_tok, batched=True, remove_columns=[c for c in ds.column_names if c not in keep])
        _update(message=f"数据集就绪：{len(ds)} 条 @ {sr}Hz", progress=12)

        accel = Accelerator(amp=False)
        device = accel.device
        loader = build_dataloader(ds, accelerator=accel,
                                  batch_size=int(cfg["batch_size"]),
                                  num_workers=0, drop_last=False)
        packer = BatchProcessor(config=tts.config, audio_vae=tts.audio_vae,
                                dataset_cnt=len(ds), device=device)

        optimizer = torch.optim.AdamW(lora_params, lr=float(cfg["lr"]), weight_decay=0.01)
        accum = max(1, int(cfg["grad_accum"]))
        epochs = max(1, int(cfg["epochs"]))
        total_steps = len(loader) * epochs
        _update(total_steps=total_steps, status="training")

        # ---------- 5. 训练循环 ----------
        gstep = 0
        running_loss, running_diff, running_stop, n_acc = 0.0, 0.0, 0.0, 0
        done = False
        for epoch in range(1, epochs + 1):
            if _stop_flag.is_set():
                break
            optimizer.zero_grad(set_to_none=True)
            for it, batch in enumerate(loader):
                if _stop_flag.is_set():
                    done = True
                    break
                packed = packer(batch)
                out = tts(
                    text_tokens=packed["text_tokens"],
                    text_mask=packed["text_mask"],
                    audio_feats=packed["audio_feats"],
                    audio_mask=packed["audio_mask"],
                    loss_mask=packed["loss_mask"],
                    position_ids=packed["position_ids"],
                    labels=packed["labels"],
                )
                diff_l = out["loss/diff"]
                stop_l = out["loss/stop"]
                # 保险：若引擎返回非标量则取均值
                diff_f = diff_l.float().mean() if diff_l.dim() else diff_l.float()
                stop_f = stop_l.float().mean() if stop_l.dim() else stop_l.float()
                loss = diff_f + stop_f
                (loss / accum).backward()
                running_loss += float(loss.item())
                running_diff += float(diff_f.item())
                running_stop += float(stop_f.item())
                n_acc += 1
                gstep += 1
                if gstep % accum == 0 or (it + 1 == len(loader)):
                    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                _update(
                    epoch=epoch, total_epochs=epochs, step=gstep,
                    progress=min(95, 12 + int(83 * gstep / max(1, total_steps))),
                    loss=round(running_loss / max(1, n_acc), 4),
                    loss_diff=round(running_diff / max(1, n_acc), 4),
                    loss_stop=round(running_stop / max(1, n_acc), 4),
                    message=f"epoch {epoch}/{epochs} · step {gstep}/{total_steps}",
                )
                if gstep % 5 == 0:
                    hist = (get_status().get("loss_history") or [])
                    hist.append([gstep, round(running_loss / max(1, n_acc), 4)])
                    _update(loss_history=hist[-400:])
            if done:
                break

        # ---------- 6. 保存 LoRA ----------
        _update(status="saving", message="保存 LoRA 权重…", progress=96)
        name = (cfg.get("lora_name") or "").strip() or f"lora_{time.strftime('%m%d_%H%M%S')}"
        out_dir = tstore.loras_dir() / name
        out_dir.mkdir(parents=True, exist_ok=True)
        sd = tts.get_lora_state_dict()
        sd = {k: v.detach().float().cpu().contiguous() for k, v in sd.items()}
        try:
            from safetensors.torch import save_file
            save_file(sd, str(out_dir / "lora_weights.safetensors"))
        except Exception:
            torch.save(sd, str(out_dir / "lora_weights.ckpt"))
        meta = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_samples": n_lines,
            "steps": gstep,
            "final_loss": round(running_loss / max(1, n_acc), 4),
            "config": {k: cfg[k] for k in ("lora_r", "lora_alpha", "lr", "epochs", "batch_size", "grad_accum")},
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        elapsed = round(time.time() - t0, 1)
        stopped = _stop_flag.is_set()
        _update(running=False, status="stopped" if stopped else "done", progress=100,
                message=("已停止并保存" if stopped else "训练完成") + f" · {name} · 用时 {elapsed}s",
                lora_name=name, finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))

    except Exception as e:
        _update(running=False, status="error", error=f"{type(e).__name__}: {e}",
                message=f"训练失败: {e}", traceback=traceback.format_exc()[-2000:],
                finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    finally:
        # 无论成败，释放训练占用的显存（推理模型会在下次请求时自动重新加载）
        try:
            del core  # noqa
        except Exception:
            pass
        try:
            import gc as _g
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _g.collect()
        except Exception:
            pass

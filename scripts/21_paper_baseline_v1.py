"""Auditable P0/P1 baseline runner for the G-MTC reproduction.

This entry point deliberately bypasses ``main_gmtc.py`` because that legacy
orchestrator enables unrelated innovations.  It provides gated phases so a
test split cannot be used while the configuration is still being selected.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.seasonal import STL


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "paper_baseline_v1"
LOGS = OUT / "training_logs"
CONFIGS = OUT / "configs"
FIGURES = ROOT / "output" / "figures"
PAPER = Path(r"C:\Users\ROG\Documents\Data-augmented SOH estimation for lithium-ion batteries under.pdf")
SEEDS = [42, 52, 62, 72, 82]
BATTERIES = ["B0007", "B0018"]
HF_COLS = ["HF1", "HF2", "HF3"]

ENGINE_SPEC = importlib.util.spec_from_file_location(
    "physics_engine", ROOT / "scripts" / "08_physics_trend_ablation.py"
)
engine = importlib.util.module_from_spec(ENGINE_SPEC)
ENGINE_SPEC.loader.exec_module(engine)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def ensure_layout() -> None:
    for path in (OUT, LOGS, CONFIGS, FIGURES):
        path.mkdir(parents=True, exist_ok=True)


def compliance_rows() -> list[dict]:
    rows = [
        ("HF1", "Paper-specified", "PDF p3 Eq.(1): constant-voltage charging duration", "scripts/01_extract_hf.py:32-39", True, False, False, "确认4.2V至20mA区间"),
        ("HF2", "Paper-specified", "PDF p3 Eq.(2): constant-current charging duration", "scripts/01_extract_hf.py:40-46", True, False, False, "代码先达到4.2V再搜索20mA；命名与PDF正文可能互换，需报告"),
        ("HF3", "Paper-specified", "PDF p3 Eq.(3): 3.8V至4.0V时间", "scripts/01_extract_hf.py:48-50", True, False, False, "无"),
        ("Hampel平滑", "Engineering-stabilization", "PDF未披露", "scripts/01_extract_hf.py:17-29", False, True, True, "center=True含未来邻点；P0数据已生成，必须披露且不作为在线因果特征"),
        ("SCF-GAN源电池", "Paper-specified", "PDF p9: NASA B0005/B0006仅用于SCF-GAN", "scripts/02_scf_gan.py:39-59", True, False, False, "B0007/B0018均未进入源训练"),
        ("SCF-GAN条件生成", "Paper-specified", "PDF p5 Eqs.(5)-(16), Tables 2-3", "scripts/02_scf_gan.py:76-187", False, True, False, "论文未披露全部层宽；当前生成3个HF并保留真实训练SOH，属工程近似"),
        ("增强倍数", "Paper-specified", "PDF p10: 5套合成数据", "scripts/02_scf_gan.py:189-227", True, False, False, "DatasetID=0真实，1-5合成"),
        ("增强边界", "Paper-specified", "PDF p9-p10: 目标训练小样本增强", "scripts/02_scf_gan.py:205-227", True, False, False, "仅目标训练前缀；验证/测试真实"),
        ("STL加性分解", "Paper-specified", "PDF p6 Eq.(17)", "scripts/08_physics_trend_ablation.py:173-184", True, False, False, "period=12是论文示例与工程选择"),
        ("STL切分边界", "Engineering-stabilization", "PDF未披露因果实现", "scripts/08_physics_trend_ablation.py:173-184", True, True, False, "先切分后仅拟合训练前缀，防止验证/测试泄漏"),
        ("Trend TimesNet", "Paper-specified", "PDF p6: deep TimesNet", "scripts/08_physics_trend_ablation.py:198-207", True, False, False, "original TimesNet"),
        ("Seasonal TimesNet", "Paper-specified", "PDF p6: lightweight TimesNet variant", "scripts/08_physics_trend_ablation.py:198-207", False, True, False, "任务预注册要求物理消融使用original TimesNet；与PDF轻量分支冲突"),
        ("Residual iTransformer", "Paper-specified", "PDF p6-p7", "scripts/03_time_series.py:138-149", True, False, False, "动态图注意力细节未披露，当前为工程近似iTransformer"),
        ("10步窗口", "Paper-specified", "PDF p6: dedicated 10-step sliding windows", "scripts/03_time_series.py:22-47", True, False, False, "窗口按DatasetID构建，不跨轨迹"),
        ("Attention Bi-LSTM输入", "Paper-specified", "PDF p7/p9 Table 4: 9d", "scripts/04_attention_fusion.py:87-103", False, True, False, "论文写预测/误差/比率；为防标签泄漏，当前用预测/增量/比率"),
        ("Attention Bi-LSTM结构", "Paper-specified", "PDF p9 Table 4", "scripts/04_attention_fusion.py:147-215", True, False, False, "BiLSTM hidden16, 32->128->64->32->3, 3->16->1"),
        ("融合权重损失", "Paper-inferred", "PDF p7 Eq.(24)未给Lweight公式", "scripts/04_attention_fusion.py:218-225", False, True, False, "L2集中度惩罚为工程近似"),
        ("统一训练配置", "Paper-specified", "PDF p9: lr=.001,batch=32,patience=50", "scripts/03_time_series.py:22-29; scripts/04_attention_fusion.py:28-39", True, False, False, "总epoch与随机种子论文未完整披露"),
        ("梯度裁剪/AdamW/调度器", "Engineering-stabilization", "PDF未披露", "scripts/03_time_series.py:152-180; scripts/04_attention_fusion.py:228-290", False, True, False, "P1稳定化，不称为论文原始方法"),
        ("轻量TimesNet/OSELM/多速率/迁移", "Legacy-innovation", "非原论文基线", "main_gmtc.py:25-40", False, False, True, "独立入口完全绕过；物理实验禁用"),
        ("CNN/LSTM/Transformer比较", "Paper-specified", "PDF p10 Table5,p14 Table6", "baseline_compare.py", False, True, True, "旧脚本仅B0018且协议不同；需新公平比较"),
        ("目标前20%小样本", "Paper-specified", "PDF p10 section4.2", "protocol.json", False, True, False, "与工程60/20/20分开，不混表"),
        ("工程60/20/20", "Engineering-stabilization", "本项目预注册协议", "scripts/21_paper_baseline_v1.py", True, True, False, "不是零标签纯跨电池"),
        ("Trend物理损失", "Legacy-innovation", "基线冻结后才启用", "scripts/08_physics_trend_ablation.py:47-56", True, False, False, "P0 mode=none,lambda=0"),
    ]
    columns = ["PaperModule", "Classification", "PaperEvidence", "CurrentCodeLocation",
               "FullyImplemented", "EngineeringApproximation", "LeakageRisk", "PendingOrNote"]
    return [dict(zip(columns, row)) for row in rows]


def audit_data() -> tuple[pd.DataFrame, dict]:
    rows = []
    gates = {
        "source_batteries_exactly_B0005_B0006": True,
        "B0007_not_source_pretraining": True,
        "raw_core_fields_complete": True,
        "augmentation_train_only": True,
        "gan_trajectories_continuous": True,
        "gan_cycles_inside_train_prefix": True,
        "stl_train_reconstruction": True,
    }
    scf_text = (ROOT / "scripts" / "02_scf_gan.py").read_text(encoding="utf-8")
    gates["source_batteries_exactly_B0005_B0006"] = 'source_bats = ["B0005", "B0006"]' in scf_text
    gates["B0007_not_source_pretraining"] = gates["source_batteries_exactly_B0005_B0006"]
    for battery in ["B0005", "B0006", "B0007", "B0018"]:
        raw = pd.read_csv(ROOT / "data" / "hf_csv" / f"{battery}_hf.csv")
        numeric = raw[HF_COLS + ["SOH"]].apply(pd.to_numeric, errors="coerce")
        missing = int(numeric.isna().any(axis=1).sum())
        gates["raw_core_fields_complete"] &= missing == 0
        n = len(numeric); train_end, val_end = int(n * .6), int(n * .8)
        reconstruction_error = np.nan
        if battery in BATTERIES:
            dec = STL(numeric["SOH"].to_numpy(float)[:train_end] * 100,
                      period=engine.ts.STL_PERIOD, robust=True).fit()
            reconstruction_error = float(np.max(np.abs(
                numeric["SOH"].to_numpy(float)[:train_end] * 100
                - (dec.trend + dec.seasonal + dec.resid)
            )))
            gates["stl_train_reconstruction"] &= reconstruction_error < 1e-6
        rows.append({
            "BatteryID": battery, "DatasetID": "raw", "Rows": n,
            "CycleStart": 0, "CycleEnd": n - 1, "MissingCoreRows": missing,
            "TrainEndExclusive": train_end, "ValidationEndExclusive": val_end,
            "STLTrainMaxReconstructionError": reconstruction_error,
            "ContinuousCycle": True, "InsideTrainPrefix": True,
        })
        if battery not in BATTERIES:
            continue
        aug = pd.read_csv(ROOT / "output" / f"{battery}_aug_train_hf.csv")
        for dataset_id, group in aug.groupby("DatasetID", sort=True):
            cycles = np.sort(group["Cycle"].to_numpy(int))
            continuous = bool(len(cycles) < 2 or np.all(np.diff(cycles) == 1))
            inside = bool(np.all((cycles >= 0) & (cycles < train_end)))
            gates["gan_trajectories_continuous"] &= continuous
            gates["gan_cycles_inside_train_prefix"] &= inside
            rows.append({
                "BatteryID": battery, "DatasetID": int(dataset_id), "Rows": len(group),
                "CycleStart": int(cycles.min()), "CycleEnd": int(cycles.max()),
                "MissingCoreRows": int(group[HF_COLS + ["SOH"]].isna().any(axis=1).sum()),
                "TrainEndExclusive": train_end, "ValidationEndExclusive": val_end,
                "STLTrainMaxReconstructionError": np.nan,
                "ContinuousCycle": continuous, "InsideTrainPrefix": inside,
            })
    return pd.DataFrame(rows), {key: bool(value) for key, value in gates.items()}


def protocol() -> dict:
    return {
        "paper_protocol": {
            "status": "specified-but-not-equivalent-to-engineering-protocol",
            "target_labeled_prefix": 0.20,
            "synthetic_trajectories_per_target": 5,
            "evidence": "PDF page 10, section 4.2",
            "unresolved": "PDF does not fully disclose a reproducible validation/test boundary for every table",
        },
        "engineering_protocol": {
            "source_scf_gan": ["B0005", "B0006"],
            "targets": ["B0007", "B0018"],
            "target_train_adaptation": 0.60,
            "validation": 0.20,
            "strict_test": 0.20,
            "zero_label_cross_battery": False,
            "target_test_status": "previously-contacted development evaluation sets",
            "confirmation_limit": "requires untouched external battery/dataset or preregistered outer holdout",
        },
        "p0": {
            "physics_mode": "none", "lambda_phy": 0.0,
            "trend_model": "original TimesNet", "seasonal_model": "original TimesNet",
            "residual_model": "iTransformer", "fusion": "Attention Bi-LSTM paper mode",
            "scf_gan_changed": False, "transfer": False, "oselm": False,
            "causal_multirate": False, "full_pinn": False, "final_soh_constraint": False,
        },
        "paper_conflict": "PDF specifies lightweight Seasonal TimesNet; task preregistration requires original TimesNet for both branches",
    }


def append_attempt(attempt_id: str, change: str, reason: str, stage: str,
                   metrics: str, gate: str, test_viewed: bool) -> None:
    path = OUT / "attempt_ledger.csv"
    columns = ["AttemptID", "Timestamp", "Change", "Reason", "Stage",
               "Metrics", "FailedGate", "TestResultsViewed"]
    if path.exists():
        frame = pd.read_csv(path)
        if attempt_id in set(frame["AttemptID"].astype(str)):
            return
    else:
        frame = pd.DataFrame(columns=columns)
    row = pd.DataFrame([{
        "AttemptID": attempt_id, "Timestamp": datetime.now().isoformat(timespec="seconds"),
        "Change": change, "Reason": reason, "Stage": stage, "Metrics": metrics,
        "FailedGate": gate, "TestResultsViewed": bool(test_viewed),
    }])
    pd.concat([frame, row], ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")


def next_attempt_id(base: str) -> str:
    path = OUT / "attempt_ledger.csv"
    if not path.exists():
        return base
    used = set(pd.read_csv(path)["AttemptID"].astype(str))
    if base not in used:
        return base
    revision = 1
    while f"{base}R{revision}" in used:
        revision += 1
    return f"{base}R{revision}"


def repair_smoke_ledger() -> None:
    """Mark pre-gate smoke rows honestly after the numerical gate was added."""
    path = OUT / "attempt_ledger.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    frame["FailedGate"] = frame["FailedGate"].astype("object")
    historical_scale_bug = frame["AttemptID"].astype(str).eq("B01R3") & frame["FailedGate"].isna()
    if historical_scale_bug.any():
        frame.loc[historical_scale_bug, "FailedGate"] = "fusion_target_not_standardized"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    changed = False
    for index, row in frame.loc[frame["Stage"].eq("smoke")].iterrows():
        if pd.notna(row.get("FailedGate")):
            continue
        try:
            metrics = pd.DataFrame(json.loads(row["Metrics"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not metrics.empty and (metrics["RMSE"] >= 50.0).any():
            frame.loc[index, "FailedGate"] = "smoke_numerical_sanity"
            changed = True
    if changed:
        frame.to_csv(path, index=False, encoding="utf-8-sig")


def mark_attempt_failed(attempt_id: str, gate: str) -> None:
    path = OUT / "attempt_ledger.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    frame["FailedGate"] = frame["FailedGate"].astype("object")
    mask = frame["AttemptID"].astype(str).eq(attempt_id)
    if mask.any():
        frame.loc[mask, "FailedGate"] = gate
        frame.to_csv(path, index=False, encoding="utf-8-sig")


def run_audit() -> None:
    ensure_layout()
    pd.DataFrame(compliance_rows()).to_csv(
        OUT / "method_compliance_matrix.csv", index=False, encoding="utf-8-sig"
    )
    data, gates = audit_data()
    data.to_csv(OUT / "data_audit.csv", index=False, encoding="utf-8-sig")
    write_json(OUT / "protocol.json", protocol())
    config = {
        "version": "paper_baseline_v1", "git_head_at_audit": git_head(),
        "paper_path": str(PAPER), "paper_sha256": sha256_file(PAPER),
        "data_gates": gates, "device": str(engine.DEVICE),
        "torch_version": torch.__version__, "seeds": SEEDS,
    }
    config["config_hash"] = config_hash(config)
    write_json(CONFIGS / "audit_config.json", config)
    lock = {
        "BaselineComplete": False, "PaperNumericMatched": False,
        "TransformerBestObserved": None, "validation_complete": False,
        "configuration_locked": False, "formal_test_completed": False,
        "config_hash": None, "data_gates": gates,
    }
    write_json(OUT / "baseline_lock.json", lock)
    append_attempt("B00", "建立方法矩阵与数据门禁", "先审计再训练", "audit",
                   json.dumps(gates, ensure_ascii=False),
                   "" if all(gates.values()) else "data_audit", False)
    if not all(gates.values()):
        raise RuntimeError(f"数据门禁失败：{gates}")
    print(json.dumps(gates, ensure_ascii=False, indent=2))


def run_seed(seed: int, split: str, epochs: int, patience: int, label: str,
             lambda_weight: float, fusion_stable_init: bool,
             use_component_lag: bool, clip_recursive_state: bool):
    engine.fusion.LAMBDA_WEIGHT = float(lambda_weight)
    run_dir = LOGS / f"{label}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows, frames, protocols = [], [], []
    for battery in BATTERIES:
        result_rows, result_frames, manifest = engine.run_battery(
            battery, ["none"], 0.0, epochs, patience, run_dir, seed,
            "test" if split == "test" else "validation",
            fusion_stable_init,
            use_component_lag,
            clip_recursive_state,
        )
        rows.extend(result_rows); frames.extend(result_frames); protocols.append(manifest)
    metrics = pd.DataFrame(rows)
    metrics.insert(1, "Seed", seed)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.insert(1, "Seed", seed)
    predictions["DatasetID"] = "raw"
    write_json(run_dir / "protocol_manifest.json", protocols)
    return metrics, predictions


def run_smoke(epochs: int, patience: int, lambda_weight: float,
              fusion_stable_init: bool, use_component_lag: bool,
              clip_recursive_state: bool) -> None:
    repair_smoke_ledger()
    attempt_id = next_attempt_id("B01")
    label = f"{attempt_id.lower()}_smoke_e{epochs}_p{patience}_lw{lambda_weight:g}"
    metrics, predictions = run_seed(
        SEEDS[0], "validation", epochs, patience, label, lambda_weight,
        fusion_stable_init,
        use_component_lag,
        clip_recursive_state,
    )
    metrics.to_csv(OUT / "smoke_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT / "smoke_predictions.csv", index=False, encoding="utf-8-sig")
    finite = bool(np.isfinite(metrics[["RMSE", "MAE", "MAPE", "R2"]].to_numpy()).all())
    numerically_sane = finite and bool((metrics["RMSE"] < 50.0).all())
    append_attempt(attempt_id, "original TimesNet严格边界冒烟", "验证形状与数值稳定",
                   "smoke", metrics.to_json(orient="records"),
                   "" if numerically_sane else "smoke_numerical_sanity", False)
    if not numerically_sane:
        raise RuntimeError("冒烟未通过数值合理性门槛")
    print(metrics.to_string(index=False))


def run_validation(epochs: int, patience: int, seed_count: int, lambda_weight: float,
                   fusion_stable_init: bool, use_component_lag: bool,
                   clip_recursive_state: bool) -> None:
    attempt_base = {1: "B02", 3: "B03", 5: "B04"}[seed_count]
    attempt_id = next_attempt_id(attempt_base)
    label = f"{attempt_id.lower()}_validation_lw{lambda_weight:g}"
    all_metrics, all_predictions = [], []
    for seed in SEEDS[:seed_count]:
        metrics, predictions = run_seed(
            seed, "validation", epochs, patience, label, lambda_weight,
            fusion_stable_init,
            use_component_lag,
            clip_recursive_state,
        )
        all_metrics.append(metrics); all_predictions.append(predictions)
    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    tag = f"{seed_count}seed_lw{lambda_weight:g}"
    metrics.to_csv(OUT / f"{attempt_id.lower()}_validation_{tag}_metrics.csv",
                   index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT / f"{attempt_id.lower()}_validation_{tag}_predictions.csv",
                       index=False, encoding="utf-8-sig")
    config = {
        "architecture": "original TimesNet + original TimesNet + iTransformer + Attention Bi-LSTM",
        "epochs": epochs, "patience": patience, "seeds": SEEDS[:seed_count],
        "physics_mode": "none", "lambda_phy": 0.0, "lambda_weight": lambda_weight,
        "fusion_stable_init": fusion_stable_init,
        "component_input": "HF+lagged-component" if use_component_lag else "HF-only",
        "clip_recursive_state": clip_recursive_state,
        "split": "60/20/20",
        "test_used_for_selection": False, "git_head": git_head(),
    }
    config["config_hash"] = config_hash(config)
    write_json(CONFIGS / f"{attempt_id.lower()}_{tag}_config.json", config)
    finite = bool(np.isfinite(metrics[["RMSE", "MAE", "MAPE", "R2"]].to_numpy()).all())
    grouped_rmse = metrics.groupby("BatteryID")["RMSE"]
    max_rmse = grouped_rmse.max()
    mean_rmse = grouped_rmse.mean().clip(lower=1e-8)
    std_rmse = grouped_rmse.std().fillna(0.0)
    no_divergence = bool((max_rmse < 10.0).all() and (std_rmse <= mean_rmse).all())
    stable = finite and (seed_count < 5 or no_divergence)
    append_attempt(attempt_id,
                   f"{seed_count}随机种子验证", "验证跨种子稳定性", "validation",
                   metrics.to_json(orient="records"),
                   "" if stable else ("multiseed_divergence" if finite else "finite_metrics"), False)
    if not stable:
        if seed_count == 5:
            lock = json.loads((OUT / "baseline_lock.json").read_text(encoding="utf-8"))
            lock.update({"validation_complete": False, "configuration_locked": False,
                         "config_hash": None, "locked_config": None})
            write_json(OUT / "baseline_lock.json", lock)
        raise RuntimeError("多随机种子验证出现非有限值或跨种子发散")
    if seed_count == 5:
        lock = json.loads((OUT / "baseline_lock.json").read_text(encoding="utf-8"))
        lock.update({"validation_complete": True, "configuration_locked": True,
                     "config_hash": config["config_hash"], "locked_config": config})
        write_json(OUT / "baseline_lock.json", lock)
        metrics.to_csv(OUT / "validation_multiseed_metrics.csv", index=False, encoding="utf-8-sig")
        predictions.to_csv(OUT / "validation_per_cycle_predictions.csv", index=False,
                           encoding="utf-8-sig")
    print(metrics.groupby("BatteryID")[["RMSE", "MAE", "MAPE", "R2"]].agg(["mean", "std", "median"]))


def run_formal_test(epochs: int, patience: int, lambda_weight: float,
                    fusion_stable_init: bool, use_component_lag: bool,
                    clip_recursive_state: bool) -> None:
    lock_path = OUT / "baseline_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not lock.get("configuration_locked") or not lock.get("validation_complete"):
        raise RuntimeError("配置尚未通过五随机种子验证并锁定，禁止访问测试集")
    if lock.get("formal_test_completed"):
        raise RuntimeError("正式测试已经执行；拒绝重复访问同一测试集")
    expected = lock["locked_config"]
    if expected["epochs"] != epochs or expected["patience"] != patience:
        raise RuntimeError("正式测试参数与锁定配置不一致")
    if expected["lambda_weight"] != lambda_weight:
        raise RuntimeError("正式测试L_weight与锁定配置不一致")
    if expected["fusion_stable_init"] != fusion_stable_init:
        raise RuntimeError("正式测试融合初始化与锁定配置不一致")
    expected_lag = expected["component_input"] == "HF+lagged-component"
    if expected_lag != use_component_lag:
        raise RuntimeError("正式测试组件输入与锁定配置不一致")
    if expected["clip_recursive_state"] != clip_recursive_state:
        raise RuntimeError("正式测试递归边界与锁定配置不一致")
    all_metrics, all_predictions = [], []
    for seed in SEEDS:
        metrics, predictions = run_seed(
            seed, "test", epochs, patience, "formal_test", lambda_weight,
            fusion_stable_init,
            use_component_lag,
            clip_recursive_state,
        )
        all_metrics.append(metrics); all_predictions.append(predictions)
    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics.to_csv(OUT / "multiseed_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT / "per_cycle_predictions.csv", index=False, encoding="utf-8-sig")
    attention = predictions.groupby(["BatteryID", "Seed"], as_index=False).agg(
        MeanWeightTrend=("w_Trend", "mean"),
        MeanWeightSeasonal=("w_Seasonal", "mean"),
        MeanWeightResidual=("w_Residual", "mean"),
        StdWeightTrend=("w_Trend", "std"),
        StdWeightSeasonal=("w_Seasonal", "std"),
        StdWeightResidual=("w_Residual", "std"),
    )
    attention["WeightSumMaxAbsError"] = predictions.groupby(["BatteryID", "Seed"]).apply(
        lambda q: float(np.max(np.abs(q[["w_Trend", "w_Seasonal", "w_Residual"]].sum(axis=1) - 1))),
        include_groups=False,
    ).to_numpy()
    attention["TrendCollapse"] = attention["MeanWeightTrend"] > .95
    attention.to_csv(OUT / "attention_diagnostics.csv", index=False, encoding="utf-8-sig")
    summary = metrics.groupby("BatteryID")["RMSE"].agg(["mean", "std", "median"])
    paper_rmse = {"B0007": np.nan, "B0018": .9123}
    numeric_match = bool(
        abs(summary.loc["B0018", "median"] - paper_rmse["B0018"]) <= .05 * paper_rmse["B0018"]
        and abs(metrics.loc[metrics.BatteryID.eq("B0018"), "R2"].median() - .8765) <= .03
    )
    lock.update({"formal_test_completed": True, "formal_test_timestamp": datetime.now().isoformat(),
                 "PaperNumericMatched": numeric_match, "BaselineComplete": False,
                 "baseline_complete_pending": ["fair comparison models", "report and figure audit"]})
    write_json(lock_path, lock)
    append_attempt("B05", "锁定配置正式测试", "一次性评估", "formal_test",
                   metrics.to_json(orient="records"), "paper_numeric" if not numeric_match else "", True)
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["audit", "smoke", "validation1", "validation3", "validation5", "formal-test"], required=True)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lambda-weight", type=float, default=.01)
    parser.add_argument("--fusion-stable-init", action="store_true")
    parser.add_argument("--hf-only-components", action="store_true")
    parser.add_argument("--clip-recursive-state", action="store_true")
    args = parser.parse_args()
    ensure_layout()
    if args.phase == "audit":
        run_audit()
    elif args.phase == "smoke":
        run_smoke(args.epochs, args.patience, args.lambda_weight, args.fusion_stable_init,
                  not args.hf_only_components, args.clip_recursive_state)
    elif args.phase == "validation1":
        run_validation(args.epochs, args.patience, 1, args.lambda_weight, args.fusion_stable_init,
                       not args.hf_only_components, args.clip_recursive_state)
    elif args.phase == "validation3":
        run_validation(args.epochs, args.patience, 3, args.lambda_weight, args.fusion_stable_init,
                       not args.hf_only_components, args.clip_recursive_state)
    elif args.phase == "validation5":
        run_validation(args.epochs, args.patience, 5, args.lambda_weight, args.fusion_stable_init,
                       not args.hf_only_components, args.clip_recursive_state)
    else:
        run_formal_test(args.epochs, args.patience, args.lambda_weight, args.fusion_stable_init,
                        not args.hf_only_components, args.clip_recursive_state)


if __name__ == "__main__":
    main()

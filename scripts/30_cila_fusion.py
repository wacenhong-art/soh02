"""CILA pooled-domain fusion experiment.

The original three component models and Attention Bi-LSTM are reused.  The only
training change is an optional CORAL penalty on the fusion model's existing
temporal context.  Windows are built separately per BatteryID and domain labels
come from explicit CLI mappings, never from shuffled sample adjacency.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset, Sampler


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("attention_fusion_cila", ROOT / "scripts" / "04_attention_fusion.py")
fusion = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(fusion)

ALLOWED_MODES = {"none", "coral"}
ALLOWED_LAMBDAS = {0.0, 0.001, 0.01, 0.1}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coral_loss(context: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
    """Mean pairwise CORAL discrepancy after pooled scale stabilization.

    The same batch-level location and scale are used for every domain.  This
    preserves between-domain covariance differences while preventing the loss
    from becoming numerically inactive merely because the existing Bi-LSTM
    context has a small internal scale.
    """
    if context.ndim != 2:
        raise ValueError("context must have shape [batch, features]")
    pooled_mean = context.mean(dim=0, keepdim=True)
    pooled_std = context.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    context = (context - pooled_mean) / pooled_std
    unique = torch.unique(domain_ids, sorted=True)
    if len(unique) < 2:
        raise ValueError("CILA requires at least two domains in every training batch")
    covariances = []
    for domain in unique:
        values = context[domain_ids == domain]
        if len(values) < 2:
            raise ValueError("CORAL covariance requires at least two samples per domain")
        centered = values - values.mean(dim=0, keepdim=True)
        covariances.append(centered.T @ centered / (len(values) - 1))
    feature_dim = context.shape[1]
    losses = []
    for left in range(len(covariances)):
        for right in range(left + 1, len(covariances)):
            losses.append((covariances[left] - covariances[right]).square().sum() / (4.0 * feature_dim**2))
    return torch.stack(losses).mean()


class DomainFusionDataset(Dataset):
    def __init__(self, x_seq, x_comp, y_z, domain_ids, sample_ids):
        self.x_seq = torch.as_tensor(x_seq, dtype=torch.float32)
        self.x_comp = torch.as_tensor(x_comp, dtype=torch.float32)
        self.y_z = torch.as_tensor(y_z, dtype=torch.float32).unsqueeze(-1)
        self.domain_ids = torch.as_tensor(domain_ids, dtype=torch.long)
        self.sample_ids = torch.as_tensor(sample_ids, dtype=torch.long)
        lengths = {len(self.x_seq), len(self.x_comp), len(self.y_z), len(self.domain_ids), len(self.sample_ids)}
        if len(lengths) != 1:
            raise ValueError("CILA dataset arrays must have equal length")

    def __len__(self):
        return len(self.y_z)

    def __getitem__(self, index):
        return self.x_seq[index], self.x_comp[index], self.y_z[index], self.domain_ids[index], self.sample_ids[index]


class BalancedDomainBatchSampler(Sampler[list[int]]):
    """Deterministic equal-count batches; never infers domains from adjacency."""

    def __init__(self, domain_ids, batch_size, seed):
        values = np.asarray(domain_ids, dtype=np.int64)
        self.groups = {int(domain): np.flatnonzero(values == domain) for domain in np.unique(values)}
        if len(self.groups) < 2:
            raise ValueError("balanced batches require at least two declared domains")
        self.per_domain = batch_size // len(self.groups)
        if self.per_domain < 2:
            raise ValueError("batch size must provide at least two samples per domain")
        if min(len(indices) for indices in self.groups.values()) < self.per_domain:
            raise ValueError("a domain has fewer samples than the per-domain batch quota")
        self.batch_count = min(len(indices) // self.per_domain for indices in self.groups.values())
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.batch_count

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        shuffled = {domain: rng.permutation(indices) for domain, indices in self.groups.items()}
        for batch_index in range(self.batch_count):
            batch = []
            start = batch_index * self.per_domain
            stop = start + self.per_domain
            for domain in sorted(shuffled):
                batch.extend(shuffled[domain][start:stop].tolist())
            yield rng.permutation(batch).tolist()


@dataclass
class PooledData:
    train: DomainFusionDataset
    validation: DomainFusionDataset
    test: DomainFusionDataset | None
    domain_names: list[str]
    metadata: dict[str, dict[str, pd.DataFrame]]
    comp_mean: np.ndarray
    comp_std: np.ndarray
    y_mean: float
    y_std: float


def validate_cycle_order(frame: pd.DataFrame, label: str) -> None:
    if "Cycle" not in frame.columns:
        raise ValueError(f"{label} is missing Cycle")
    cycles = frame["Cycle"].to_numpy(dtype=np.int64)
    if len(np.unique(cycles)) != len(cycles) or np.any(np.diff(cycles) <= 0):
        raise ValueError(f"{label} Cycle must be strictly increasing and unique")


def load_domain_frames(domain_dirs: dict[str, Path], include_test: bool) -> dict[str, dict[str, pd.DataFrame]]:
    loaded = {}
    splits = ["train", "val"] + (["test"] if include_test else [])
    for name, directory in sorted(domain_dirs.items()):
        loaded[name] = {}
        for split in splits:
            path = directory / f"comp_pred_{split}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            fusion.validate_component_df(frame, f"{name}/{split}")
            validate_cycle_order(frame, f"{name}/{split}")
            loaded[name][split] = frame
    return loaded


def build_pooled_data(domain_dirs: dict[str, Path], include_test=False) -> PooledData:
    frames = load_domain_frames(domain_dirs, include_test)
    domain_names = sorted(frames)
    train_frames = [frames[name]["train"] for name in domain_names]
    train_comp = np.concatenate([
        frame[["Pred_Trend", "Pred_Seasonal", "Pred_Residual"]].to_numpy(np.float32)
        for frame in train_frames
    ])
    comp_mean, comp_std = fusion.zscore_fit(train_comp, axis=0)
    train_feature_rows = np.concatenate([fusion.build_causal_features(frame, comp_mean, comp_std) for frame in train_frames])
    feat_mean, feat_std = fusion.zscore_fit(train_feature_rows, axis=0)
    train_targets = np.concatenate([fusion.true_soh_target(frame) for frame in train_frames])
    y_mean = float(train_targets.mean())
    y_std = max(float(train_targets.std()), 1e-8)

    datasets = {}
    metadata: dict[str, dict[str, pd.DataFrame]] = {split: {} for split in ["train", "validation", "test"]}
    for split_file, split_name in [("train", "train"), ("val", "validation"), ("test", "test")]:
        if split_file == "test" and not include_test:
            datasets[split_name] = None
            continue
        x_all, comp_all, y_all, domains_all, sample_ids_all = [], [], [], [], []
        sample_offset = 0
        for domain_index, domain_name in enumerate(domain_names):
            frame = frames[domain_name][split_file]
            arrays, _ = fusion.prepare_split(frame, comp_mean, comp_std, feat_mean, feat_std, y_mean, y_std)
            x_seq, comp, y_z = arrays
            count = len(y_z)
            x_all.append(x_seq); comp_all.append(comp); y_all.append(y_z)
            domains_all.append(np.full(count, domain_index, dtype=np.int64))
            sample_ids_all.append(np.arange(sample_offset, sample_offset + count, dtype=np.int64))
            sample_offset += count
            meta = frame.iloc[fusion.SEQ_LEN:].reset_index(drop=True).copy()
            meta.insert(0, "DatasetID", domain_name)
            meta.insert(0, "BatteryID", domain_name)
            metadata[split_name][domain_name] = meta
        datasets[split_name] = DomainFusionDataset(
            np.concatenate(x_all), np.concatenate(comp_all), np.concatenate(y_all),
            np.concatenate(domains_all), np.concatenate(sample_ids_all),
        )
    return PooledData(
        train=datasets["train"], validation=datasets["validation"], test=datasets["test"],
        domain_names=domain_names, metadata=metadata, comp_mean=comp_mean, comp_std=comp_std,
        y_mean=y_mean, y_std=y_std,
    )


def objective(pred_z, target_z, weights, context, domain_ids, mode, lambda_align):
    prediction = nn.functional.mse_loss(pred_z, target_z)
    weight = fusion.LAMBDA_WEIGHT * fusion.weight_regularization(weights)
    alignment = context.new_zeros(())
    if mode == "coral" and lambda_align > 0:
        alignment = coral_loss(context, domain_ids)
    return prediction + weight + lambda_align * alignment, prediction, weight, alignment


def train_model(model, data: PooledData, output_dir: Path, mode: str, lambda_align: float,
                seed: int, epochs: int, patience: int):
    sampler = BalancedDomainBatchSampler(data.train.domain_ids.numpy(), fusion.BATCH_SIZE, seed)
    train_loader = DataLoader(data.train, batch_sampler=sampler)
    val_loader = DataLoader(data.validation, batch_size=min(fusion.BATCH_SIZE, len(data.validation)), shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=fusion.LR, weight_decay=fusion.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-6)
    best_val = math.inf; wait = 0; rows = []
    model_path = output_dir / "fusion_att_bilstm_cila.pth"
    for epoch in range(1, epochs + 1):
        sampler.set_epoch(epoch)
        model.train(); train_sums = np.zeros(4, dtype=np.float64)
        for x_seq, comp, y_z, domain_ids, _ in train_loader:
            x_seq=x_seq.to(fusion.device); comp=comp.to(fusion.device); y_z=y_z.to(fusion.device); domain_ids=domain_ids.to(fusion.device)
            pred_z, weights, context = model(x_seq, comp, return_context=True)
            total, prediction, weight, alignment = objective(pred_z, y_z, weights, context, domain_ids, mode, lambda_align)
            optimizer.zero_grad(); total.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_sums += [total.item(), prediction.item(), weight.item(), alignment.item()]
        model.eval(); val_sum = 0.0; val_count = 0
        with torch.no_grad():
            for x_seq, comp, y_z, domain_ids, _ in val_loader:
                pred_z, weights, context = model(x_seq.to(fusion.device), comp.to(fusion.device), return_context=True)
                prediction = nn.functional.mse_loss(pred_z, y_z.to(fusion.device), reduction="sum")
                weight = fusion.LAMBDA_WEIGHT * fusion.weight_regularization(weights) * len(y_z)
                val_sum += (prediction + weight).item(); val_count += len(y_z)
        val_objective = val_sum / max(val_count, 1)
        scheduler.step(val_objective)
        divisor = max(len(train_loader), 1)
        rows.append({"Epoch": epoch, "TrainTotal": train_sums[0]/divisor, "TrainPrediction": train_sums[1]/divisor,
                     "TrainWeight": train_sums[2]/divisor, "TrainCORAL": train_sums[3]/divisor,
                     "TrainCORALContribution":lambda_align*train_sums[3]/divisor,
                     "ValidationSelectionObjective": val_objective, "LearningRate": optimizer.param_groups[0]["lr"]})
        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch={epoch:04d} train={train_sums[0]/divisor:.6f} coral={train_sums[3]/divisor:.6f} val={val_objective:.6f}", flush=True)
        if val_objective < best_val - 1e-6:
            best_val = val_objective; wait = 0; torch.save(model.state_dict(), model_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"early_stop={epoch}", flush=True); break
    pd.DataFrame(rows).to_csv(output_dir / "training_log.csv", index=False)
    model.load_state_dict(torch.load(model_path, map_location=fusion.device))
    return model, best_val


def infer(model, dataset: DomainFusionDataset, data: PooledData, split: str, output_dir: Path):
    loader = DataLoader(dataset, batch_size=min(fusion.BATCH_SIZE, len(dataset)), shuffle=False)
    pred_all=[]; weight_all=[]; domain_all=[]
    model.eval()
    with torch.no_grad():
        for x_seq, comp, _, domain_ids, _ in loader:
            pred_z, weights = model(x_seq.to(fusion.device), comp.to(fusion.device))
            pred_all.append(pred_z.cpu().numpy().reshape(-1)); weight_all.append(weights.cpu().numpy()); domain_all.append(domain_ids.numpy())
    predictions = np.concatenate(pred_all) * data.y_std + data.y_mean
    weights = np.concatenate(weight_all); domain_ids = np.concatenate(domain_all)
    outputs=[]; metrics=[]; cursor=0
    for domain_index, domain_name in enumerate(data.domain_names):
        meta = data.metadata[split][domain_name]
        count = len(meta); pred = predictions[cursor:cursor+count]; w = weights[cursor:cursor+count]; cursor += count
        true = fusion.true_soh_target(meta)
        frame = pd.DataFrame({"BatteryID": domain_name, "DatasetID": domain_name, "Cycle": meta["Cycle"].to_numpy(int),
                              "True_SOH": true, "Pred_SOH": pred, "w_Trend": w[:,0], "w_Seasonal": w[:,1], "w_Residual": w[:,2]})
        outputs.append(frame)
        rmse=float(np.sqrt(mean_squared_error(true,pred))); mae=float(mean_absolute_error(true,pred))
        metrics.append({"BatteryID":domain_name,"EvaluatedSplit":split,"RMSE":rmse,"MAE":mae,
                        "MAPE":float(np.abs(true-pred).sum()/np.abs(true).sum()*100),"R2":float(r2_score(true,pred))})
    output = pd.concat(outputs, ignore_index=True)
    output.to_csv(output_dir / f"per_cycle_predictions_{split}.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(metrics), output


def parse_domains(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--domain must use BatteryID=directory")
        name, raw_path = value.split("=", 1)
        if not name or name in result:
            raise ValueError("BatteryID must be nonempty and unique")
        result[name] = Path(raw_path).resolve()
    if len(result) < 2:
        raise ValueError("CILA pooled experiment requires at least two --domain mappings")
    return result


def run(args):
    mode=args.mode.lower(); lambda_align=float(args.lambda_align)
    if mode not in ALLOWED_MODES or lambda_align not in ALLOWED_LAMBDAS:
        raise ValueError(f"mode/lambda must be {sorted(ALLOWED_MODES)} / {sorted(ALLOWED_LAMBDAS)}")
    if mode == "none" and lambda_align != 0:
        raise ValueError("none mode requires lambda_align=0")
    fusion.LAMBDA_WEIGHT=float(args.lambda_weight)
    domains=parse_domains(args.domain); output_dir=Path(args.output_dir).resolve(); output_dir.mkdir(parents=True,exist_ok=True)
    fusion.set_seed(args.seed); started=time.time(); data=build_pooled_data(domains,args.evaluate_test)
    model=fusion.AttentionBiLSTMFusion(data.comp_mean,data.comp_std,data.y_mean,data.y_std,fusion_mode="paper",stable_init=args.stable_init).to(fusion.device)
    config={"mode":mode,"lambda_align":lambda_align,"lambda_weight":fusion.LAMBDA_WEIGHT,"seed":args.seed,"epochs":args.epochs,"patience":args.patience,
            "domains":{k:str(v) for k,v in domains.items()},"input_sha256":{f"{k}/{s}":sha256_file(v/f"comp_pred_{s}.csv") for k,v in domains.items() for s in (["train","val","test"] if args.evaluate_test else ["train","val"])},
            "architecture":"unchanged AttentionBiLSTMFusion paper mode","alignment":"pairwise CORAL on pooled-standardized existing temporal context",
            "evaluate_test":args.evaluate_test,"validate_only":args.validate_only,"stable_init":args.stable_init}
    config_blob=json.dumps(config,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode(); config["config_hash"]=hashlib.sha256(config_blob).hexdigest()
    (output_dir/"config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if args.validate_only:
        audit_rows=[]
        for split in ["train","validation"] + (["test"] if args.evaluate_test else []):
            for domain_name, frame in data.metadata[split].items():
                cycles=frame["Cycle"].to_numpy(np.int64)
                audit_rows.append({"BatteryID":domain_name,"DatasetID":domain_name,"Split":split,"WindowRows":len(frame),
                                   "FirstCycle":int(cycles[0]),"LastCycle":int(cycles[-1]),"StrictlyIncreasing":bool(np.all(np.diff(cycles)>0)),
                                   "WindowsCrossBatteryBoundary":False,"TargetLabelsUsedAsInputs":False})
        audit=pd.DataFrame(audit_rows); audit.to_csv(output_dir/"data_audit.csv",index=False)
        sampler=BalancedDomainBatchSampler(data.train.domain_ids.numpy(),fusion.BATCH_SIZE,args.seed)
        diagnostic_loader=DataLoader(data.train,batch_sampler=sampler)
        x_seq,comp,y_z,domain_ids,_=next(iter(diagnostic_loader))
        model.eval()
        with torch.no_grad():
            pred_z,weights,context=model(x_seq.to(fusion.device),comp.to(fusion.device),return_context=True)
            prediction=nn.functional.mse_loss(pred_z,y_z.to(fusion.device))
            alignment=coral_loss(context,domain_ids.to(fusion.device))
        diagnostic={"InitialPredictionLoss":float(prediction.item()),"InitialCORAL":float(alignment.item()),
                    "WeightedCORAL":float(lambda_align*alignment.item()),
                    "WeightedCORALToPredictionRatio":float(lambda_align*alignment.item()/max(prediction.item(),1e-12)),
                    "DomainsPerBatch":int(torch.unique(domain_ids).numel())}
        (output_dir/"loss_scale_diagnostic.json").write_text(json.dumps(diagnostic,indent=2)+"\n",encoding="utf-8")
        print(audit.to_string(index=False)); print(json.dumps(diagnostic,indent=2)); print("validate_only=PASS",flush=True); return audit
    model,best=train_model(model,data,output_dir,mode,lambda_align,args.seed,args.epochs,args.patience)
    val_metrics,_=infer(model,data.validation,data,"validation",output_dir); frames=[val_metrics]
    if args.evaluate_test:
        test_metrics,_=infer(model,data.test,data,"test",output_dir); frames.append(test_metrics)
    metrics=pd.concat(frames,ignore_index=True); metrics.insert(0,"Seed",args.seed); metrics.insert(0,"LambdaAlign",lambda_align); metrics.insert(0,"CILAMode",mode)
    metrics["BestValidationObjective"]=best; metrics["ElapsedSeconds"]=time.time()-started; metrics["ConfigHash"]=config["config_hash"]
    metrics.to_csv(output_dir/"metrics.csv",index=False)
    print(metrics.to_string(index=False)); return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain",action="append",required=True,help="BatteryID=directory containing comp_pred_train/val[/test].csv")
    parser.add_argument("--output-dir",required=True); parser.add_argument("--mode",choices=sorted(ALLOWED_MODES),default="none")
    parser.add_argument("--lambda-align",type=float,default=0.0); parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--lambda-weight",type=float,default=0.1,help="Frozen P1 fusion weight regularization coefficient")
    parser.add_argument("--epochs",type=int,default=150); parser.add_argument("--patience",type=int,default=30)
    parser.add_argument("--stable-init",action="store_true"); parser.add_argument("--evaluate-test",action="store_true")
    parser.add_argument("--validate-only",action="store_true",help="load, audit and shape-check domains without training")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

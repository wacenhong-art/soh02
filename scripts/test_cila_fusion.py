"""Unit-level gates for CILA; no battery training or test metric access."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cila", ROOT / "scripts" / "30_cila_fusion.py")
cila = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = cila
SPEC.loader.exec_module(cila)


def main():
    torch.manual_seed(7)
    base = torch.randn(8, 6)
    context = torch.cat([base, base], dim=0).requires_grad_(True)
    domains = torch.tensor([0] * 8 + [1] * 8)
    zero = cila.coral_loss(context, domains)
    assert torch.allclose(zero, torch.tensor(0.0), atol=1e-10)

    changed = torch.cat([base, base * torch.tensor([1., 2., 3., 1., 2., 3.])], dim=0).requires_grad_(True)
    positive = cila.coral_loss(changed, domains)
    assert positive.item() > 0
    rescaled = cila.coral_loss(changed * 2.0 + 7.0, domains)
    assert torch.allclose(positive, rescaled, rtol=1e-5, atol=1e-7)
    positive.backward()
    assert changed.grad is not None and torch.isfinite(changed.grad).all()

    sampler = cila.BalancedDomainBatchSampler(np.array([0] * 12 + [1] * 10), batch_size=8, seed=42)
    for batch in sampler:
        batch_domains = np.array([0] * 12 + [1] * 10)[batch]
        unique, counts = np.unique(batch_domains, return_counts=True)
        assert unique.tolist() == [0, 1] and counts.tolist() == [4, 4]

    comp_mean=np.zeros(3,np.float32); comp_std=np.ones(3,np.float32)
    model=cila.fusion.AttentionBiLSTMFusion(comp_mean,comp_std,0.0,1.0)
    model.eval()
    x=torch.randn(8,cila.fusion.SEQ_LEN,9); comp=torch.randn(8,3); y=torch.randn(8,1)
    two=model(x,comp); three=model(x,comp,return_context=True)
    assert len(two)==2 and len(three)==3
    assert torch.equal(two[0],three[0]) and torch.equal(two[1],three[1]) and three[2].shape==(8,32)
    total, pred, weight, align=cila.objective(three[0],y,three[1],three[2],torch.tensor([0]*4+[1]*4),"coral",0.0)
    assert torch.equal(total,pred+weight) and align.item()==0.0

    seq=np.arange(15*2,dtype=np.float32).reshape(15,2); comps=np.zeros((15,3),np.float32); target=np.arange(15,dtype=np.float32)
    first=cila.fusion.sliding_windows(seq,comps,target,10)[0]
    second=cila.fusion.sliding_windows(seq+1000,comps,target,10)[0]
    pooled=np.concatenate([first,second])
    assert np.all(pooled[:len(first)]<1000) and np.all(pooled[len(first):]>=1000)
    print("CILA unit gates PASS: zero/positive/scale-invariant/gradient/balanced-batch/API/lambda0/window-boundary")


if __name__ == "__main__":
    main()

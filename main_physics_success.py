"""Reproduce the validation-approved successful Physics+frozen-ELM experiment."""
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"output"/"physics_success_reproduction"


def run(*args):
    subprocess.run([sys.executable,"-X","utf8",*map(str,args)],cwd=ROOT,check=True)


def main():
    run("scripts/08_physics_trend_ablation.py","--modes","none,monotonic","--epochs","150",
        "--patience","30","--lambda-phy","0.01","--seed","42","--output-dir",OUT)
    for battery in ["B0007","B0018"]:
        run("scripts/05_oselm_correction.py","--input-dir",OUT/battery/"monotonic",
            "--output-dir",OUT/"frozen_elm"/battery,"--model-name","G-MTC+monotonic")
    print(f"Successful reproduction written to {OUT}")


if __name__=="__main__":main()

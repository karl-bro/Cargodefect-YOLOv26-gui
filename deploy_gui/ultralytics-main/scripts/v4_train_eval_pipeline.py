#!/usr/bin/env python3
"""Train remaining v4 variants + evaluate + threshold sweep + select best model."""
import os, sys, csv, yaml
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO

V4_RUNS = ROOT / "runs/detect/runs/cargodefect"
DATA_YAML = "ultralytics/cfg/datasets/cargodefect-fusion.yaml"
BASE_YAML = "ultralytics/cfg/models/26/cargodefect-yolov26.yaml"
CONFIG_DIR = ROOT / "configs/ablation/quality_loss"
OUT_DIR = ROOT / "results/v4_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    {"name": "fusion_v4_pos1.0", "quality_pos_weight": 1.0},
    {"name": "fusion_v4_pos1.2", "quality_pos_weight": 1.2},
    {"name": "fusion_v4_pos1.5", "quality_pos_weight": 1.5},
    {"name": "fusion_v4_focal0.25", "quality_focal_gamma": 2.0, "quality_focal_alpha": 0.25},
    {"name": "fusion_v4_focal0.35", "quality_focal_gamma": 2.0, "quality_focal_alpha": 0.35},
]

TRAIN_ARGS = dict(
    data=DATA_YAML, epochs=100, imgsz=640, device=0, workers=4,
    project="runs/cargodefect", exist_ok=True, amp=True, cos_lr=True,
    grad_clip=1.0, optimizer="auto", lr0=0.01, lrf=0.01, momentum=0.937,
    weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8,
    warmup_bias_lr=0.1, close_mosaic=10, patience=100, seed=0,
    deterministic=True, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, cutmix=0.0,
    copy_paste=0.0, auto_augment="randaugment", erasing=0.4,
)

with open(BASE_YAML) as f:
    base_cfg = yaml.safe_load(f)

def train_variant(v):
    name = v["name"]
    wp = V4_RUNS / name / "weights" / "best.pt"
    if wp.exists():
        print(f"  [{name}] done, skip")
        return True
    cfg = deepcopy(base_cfg)
    for k, val in v.items():
        if k == "name": continue
        cfg[k] = val
    if cfg.get("quality_focal_gamma", 0) > 0:
        cfg.pop("quality_pos_weight", None)
    else:
        cfg.pop("quality_focal_gamma", None); cfg.pop("quality_focal_alpha", None)
    yp = CONFIG_DIR / f"{name}.yaml"
    with open(yp, "w") as f: yaml.dump(cfg, f, default_flow_style=False)
    batch = 8
    while batch >= 1:
        try:
            print(f"\n=== {name} batch={batch} ===")
            m = YOLO(str(yp)); m.train(batch=batch, name=name, **TRAIN_ARGS)
            return True
        except RuntimeError as e:
            if "memory" not in str(e).lower() and "alloc" not in str(e).lower(): raise
            del m; torch.cuda.empty_cache(); torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(); batch //= 2
            if batch < 1: return False
            print(f"  OOM, retry batch={batch}")
    return False

def evaluate_quality(wpath, name):
    print(f"\n  Eval {name}...")
    yolo = YOLO(str(wpath)); model = yolo.model; model.eval()
    device = next(model.parameters()).device
    data_cfg = yaml.safe_load(open(DATA_YAML))
    from ultralytics.data.cargodefect import CargoDefectDataset
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.data.augment import LetterBox
    hyp = deepcopy(DEFAULT_CFG)
    try:
        ds = CargoDefectDataset(
            img_path=data_cfg.get("val", data_cfg.get("path","")+"/images/Validation"),
            imgsz=640, batch_size=1, augment=False, hyp=hyp,
            data=data_cfg, stride=32, prefix="val")
    except Exception as e:
        print(f"  DS err: {e}"); return None
    lb = LetterBox(640, auto=False, stride=32)
    all_l, all_t = [], []
    for i in range(len(ds)):
        try: batch = ds[i]
        except: break
        if isinstance(batch["img"], torch.Tensor):
            img = batch["img"].float().unsqueeze(0).to(device)
            if img.ndim==4 and img.shape[1]==3: img = img/255.0
        else: continue
        ql = int(batch.get("quality_label", torch.tensor(0)).item()) if "quality_label" in batch else 0
        with torch.no_grad():
            out = model(img)
        aux = out[1] if isinstance(out,tuple) and len(out)==2 else (out if isinstance(out,dict) else {})
        q = aux.get("quality", {})
        if isinstance(q, dict) and "logits" in q:
            lt = q["logits"]
            if lt.numel():
                prob = float(torch.sigmoid(lt[:,0]).item())
                all_l.append(prob); all_t.append(ql)
        if (i+1)%200==0: print(f"    {i+1}")
    if not all_l:
        print("  No quality data")
        return {"name":name,"accuracy":0,"ok_accuracy":0,"ng_recall":0,"false_ok_rate":1,"false_ng_rate":0,"f1":0,"confusion":[[0,0],[0,0]],"logits":[],"targets":[]}
    la = np.array(all_l); ta = np.array(all_t)
    pred = (la>=0.5).astype(int)
    tp = int(((pred==1)&(ta==1)).sum()); tn = int(((pred==0)&(ta==0)).sum())
    fp = int(((pred==1)&(ta==0)).sum()); fn = int(((pred==0)&(ta==1)).sum())
    no = max((ta==0).sum(),1); nn = max((ta==1).sum(),1); nt = len(ta)
    acc = (tp+tn)/max(nt,1); oa = tn/no; nr = tp/nn
    fok = fn/nn; fng = fp/no; prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
    f1v = 2*prec*rec/max(prec+rec,1e-9)
    return {"name":name,"accuracy":round(acc,4),"ok_accuracy":round(oa,4),
            "ng_recall":round(nr,4),"false_ok_rate":round(fok,4),
            "false_ng_rate":round(fng,4),"f1":round(f1v,4),
            "n_total":nt,"n_ok":int((ta==0).sum()),"n_ng":int((ta==1).sum()),
            "confusion":[[tn,fp],[fn,tp]],"logits":la.tolist(),"targets":ta.tolist()}

def sweep(name, la, ta):
    la=np.array(la); ta=np.array(ta); rows=[]
    for thr_i in range(10,96,5):
        thr=thr_i/100.0; pred=(la>=thr).astype(int)
        tp=int(((pred==1)&(ta==1)).sum()); tn=int(((pred==0)&(ta==0)).sum())
        fp=int(((pred==1)&(ta==0)).sum()); fn=int(((pred==0)&(ta==1)).sum())
        no=max((ta==0).sum(),1); nn=max((ta==1).sum(),1)
        oa=tn/no; nr=tp/nn; a=(tp+tn)/len(ta); fo=fn/nn; fng=fp/no
        p=tp/max(tp+fp,1); r=tp/max(tp+fn,1); f=2*p*r/max(p+r,1e-9)
        rows.append([thr,round(a,4),round(oa,4),round(nr,4),round(fo,4),round(fng,4),round(p,4),round(r,4),round(f,4)])
    return rows

def main():
    if not torch.cuda.is_available():
        print("No CUDA"); sys.exit(1)
    print("GPU:", torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()

    # Phase 1: Train remaining
    for v in VARIANTS[1:]:
        train_variant(v)

    # Phase 2: Evaluate
    evs=[]
    for v in VARIANTS:
        wp = V4_RUNS / v["name"] / "weights" / "best.pt"
        if not wp.exists(): continue
        ev = evaluate_quality(str(wp), v["name"])
        if ev: evs.append(ev); print(f"  [{ev['name']}] Acc={ev['accuracy']} OK={ev['ok_accuracy']} NG={ev['ng_recall']} F1={ev['f1']}")

    with open(OUT_DIR/"quality_evaluation.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["name","accuracy","ok_accuracy","ng_recall","false_ok_rate","false_ng_rate","f1","n_total","n_ok","n_ng","tn","fp","fn","tp"])
        for ev in evs:
            c=ev["confusion"]; w.writerow([ev["name"],ev["accuracy"],ev["ok_accuracy"],ev["ng_recall"],ev["false_ok_rate"],ev["false_ng_rate"],ev["f1"],ev["n_total"],ev["n_ok"],ev["n_ng"],c[0][0],c[0][1],c[1][0],c[1][1]])

    # Phase 3: Sweep
    for ev in evs:
        if not ev.get("logits"): continue
        rows = sweep(ev["name"], ev["logits"], ev["targets"])
        sc = OUT_DIR / f"threshold_sweep_{ev['name']}.csv"
        with open(sc,"w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["threshold","accuracy","ok_accuracy","ng_recall","false_ok_rate","false_ng_rate","precision","recall","f1"])
            w.writerows(rows)
        best = None
        for r in rows:
            if r[4]<0.1 and r[3]>=0.85 and r[2]>=0.75 and r[1]>=0.80:
                if best is None or r[8]>best[8]: best=r
        if best is None: best=max(rows, key=lambda r:r[8])
        (OUT_DIR/f"best_threshold_{ev['name']}.md").write_text(f"# {ev['name']}\nBest threshold: {best[0]} acc={best[1]} ok_acc={best[2]} ng_recall={best[3]} f1={best[8]}\n")
        print(f"  [{ev['name']}] best_thr={best[0]} F1={best[8]:.4f}")

    # Phase 4: Select best
    scd=[]
    for ev in evs:
        s=0
        if ev["ng_recall"]>=0.85: s+=3
        if ev["ok_accuracy"]>=0.75: s+=3
        if ev["accuracy"]>=0.80: s+=2
        s+=(1-ev["false_ok_rate"])*3+(1-ev["false_ng_rate"])*2
        scd.append((s,ev))
    scd.sort(reverse=True)
    best=scd[0][1]
    bp = V4_RUNS / best["name"] / "weights" / "best.pt"
    lines=["# v4 Final Report","","|Name|Score|Acc|OKAcc|NGRec|FOK|FNG|F1|","|---|---|---|---|---|---|---|---|---|"]
    for s,ev in scd: lines.append(f"|{ev['name']}|{s:.1f}|{ev['accuracy']}|{ev['ok_accuracy']}|{ev['ng_recall']}|{ev['false_ok_rate']}|{ev['false_ng_rate']}|{ev['f1']}|")
    lines+=["",f"## Best: {best['name']}",f"Path: {bp}"]
    (OUT_DIR/"v4_final_report.md").write_text("\n".join(lines))
    print(f"\nBest: {best['name']}\nReport: {OUT_DIR/'v4_final_report.md'}")

if __name__=="__main__":
    main()

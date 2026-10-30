"""TEST EVALUATION: off by default, opened only by an explicit decision (preregistration: test is evaluated once, with
the final selected settings only). Measures the final model of finished runs on the official test set (1,561) and on the
clean test set (images without a near-duplicate in train or dev).
Usage: python evaluate_test.py --root <drive> --target /content/plantseg --runs a_v3L_lora_s0,b_cnxL_s0 --yes_test"""
import argparse, json, os, subprocess, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from run_queue import Queue, now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--target', required=True)
    ap.add_argument('--runs', default='', help='comma-separated id list; if empty, all finished runs')
    ap.add_argument('--yes_test', action='store_true', help='deliberate confirmation flag to open the test set')
    ap.add_argument('--zip', default=None)
    a = ap.parse_args()
    if not a.yes_test:
        raise SystemExit('Test is closed. Pass --yes_test only by explicit decision.')
    data = f'{a.target}/prep'
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.py'), '--target', a.target, '--test']
    if a.zip:
        cmd += ['--zip', a.zip]
    subprocess.run(cmd, check=True)
    split = json.load(open(f'{data}/split.json'))
    lk = json.load(open(f'{data}/leakage.json'))
    clean = set(lk['clean_test']); clean_tr = set(lk['clean_test_train_only'])
    Q = Queue(a.root)
    d = Q.read()
    ids = [x for x in a.runs.split(',') if x] or [k['id'] for k in d['runs'] if k['status'] == 'done']
    out_dir = os.path.join(a.root, 'results', 'test'); os.makedirs(out_dir, exist_ok=True)
    dev = 'cuda'; dtype = common.amp_dtype()
    for rid in ids:
        if os.path.exists(os.path.join(out_dir, f'{rid}.json')):
            print('test result exists, skipping:', rid); continue  # a rerun must not open the test a second time
        mp = os.path.join(a.root, 'model', f'{rid}.pt')
        if not os.path.exists(mp):
            print('no model, skipping:', rid); continue
        ck = torch.load(mp, map_location='cpu', weights_only=False)
        k, r = ck['run'], ck['recipe']
        model = common.Seg(k['bb'], k['mode'], head_ch=r['head_ch'], aux=r['aux'], drop_path=r['drop_path'])
        missing, unexpected = model.load_state_dict(ck['model'], strict=False)
        P = dict(model.named_parameters())
        assert not unexpected and all(not P[m].requires_grad for m in missing if m in P)
        model = model.to(dev)
        stems = split['test']
        res, iou, acc, per = common.evaluate(model, stems, dev, data, 'test',
                                             out_npz=os.path.join(out_dir, f'{rid}_test_per_image.npz'), dtype=dtype)
        sub = common.subset_report(per, stems, data, clean)
        cl2 = np.array([s in clean_tr for s in stems])
        names = common.class_names(data)
        res.update({'id': rid, 'split': 'test (official)', **sub, 'mIoU_clean_train_only': round(common.miou_subset(per[cl2]), 2),
                    'clean_definition': lk['clean_test_definition'],
                    'class_IoU': {names[c]: (None if np.isnan(iou[c]) else round(float(iou[c]) * 100, 3)) for c in range(common.NC)},
                    'timestamp': now().strftime('%F %H:%M')})
        json.dump(res, open(os.path.join(out_dir, f'{rid}.json'), 'w'), indent=1, ensure_ascii=False)
        print(f"TEST {rid}: mIoU {res['mIoU']:.2f} clean {sub['mIoU_clean']} (n={sub['n_clean']}) mAcc {res['mAcc']:.2f}", flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

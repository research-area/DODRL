#!/usr/bin/env python3
"""Choose next Approach C weights from a completed warmstart batch.

Objective: maximize mean energy saving vs ondemand, subject to service-time
degradation <= 10% (agent not slower than +10% vs ondemand).

Pairs each evaluation window against the interleaved ondemand reference
(phase 'reference_interleaved'), which is the true eval-time baseline.
Usage: choose_weights_warmstart.py <results_root> <current_e> <current_l>
Prints "<next_e> <next_l>" then human summary. 'no-data' if nothing usable.
"""
import argparse, csv, glob, os, statistics

def mean(xs): return sum(xs)/len(xs) if xs else 0.0

def gather(root):
    # per lambda: eval vs interleaved-ondemand, and eval vs perf
    by = {}
    for r in glob.glob(os.path.join(root, '**', 'runs.csv'), recursive=True):
        base = os.path.basename(os.path.dirname(r))
        lam = base.split('-l')[-1] if '-l' in base else os.path.basename(root)
        ev=[]; intr=[]; perf=[]
        with open(r) as f:
            for row in csv.DictReader(f):
                p = row['phase']
                try:
                    svc=float(row['mean_service_time_s']); en=float(row['domain_energy_j'])
                except Exception: continue
                rec=(svc,en)
                if p=='evaluation': ev.append(rec)
                elif p=='reference_interleaved': intr.append(rec)
                elif p=='performance_reference': perf.append(rec)
        if ev and intr and perf:
            by[lam]=(ev,intr,perf)
    return by

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('root'); ap.add_argument('cur_e',type=float,nargs='?',default=0.5)
    ap.add_argument('cur_l',type=float,nargs='?',default=0.5)
    a=ap.parse_args()
    by=gather(a.root)
    if not by:
        print('no-data'); print('no usable runs.csv'); return
    pairs_od=[]; pairs_pf=[]
    for lam,(ev,intr,perf) in sorted(by.items()):
        n=min(len(ev),len(intr)); m=min(n,len(perf))
        if n<1: continue
        en_a=mean([x[1] for x in ev[:n]]); svc_a=mean([x[0] for x in ev[:n]])
        en_od=mean([x[1] for x in intr[:n]]); svc_od=mean([x[0] for x in intr[:n]])
        en_pf=mean([x[1] for x in perf[:m]]); svc_pf=mean([x[0] for x in perf[:m]])
        pairs_od.append((100*(en_a/en_od-1), 100*(svc_a/svc_od-1)))  # >0 = uses more / slower
        pairs_pf.append((100*(en_a/en_pf-1), 100*(svc_a/svc_pf-1)))
    en_od=mean([e for e,_ in pairs_od]); svc_od=mean([s for _,s in pairs_od])
    en_pf=mean([e for e,_ in pairs_pf]); svc_pf=mean([s for _,s in pairs_pf])
    # energy SAVING = -en_od (positive when agent uses less energy)
    energy_save = -en_od
    svc_deg = svc_od                       # positive = agent slower

    e,l = a.cur_e, a.cur_l
    reason=''
    if svc_deg > 10.0:
        # unacceptable slowdown -> push latency weight up (protect performance)
        new_l=min(0.95, l+0.15); reason=f'svc degradation {svc_deg:.1f}% > 10% -> shift to latency'
    elif svc_deg < 2.0 and energy_save < 5.0:
        # capacity headroom and weak saving -> push energy harder
        new_l=max(0.05, l-0.15); reason=f'svc ok ({svc_deg:.1f}%) but low energy save ({energy_save:.1f}%) -> push energy'
    elif svc_deg < 5.0 and energy_save >= 5.0:
        # within budget and good saving -> keep, tiny nudge to energy if headroom
        new_l=max(0.05, l-0.05); reason=f'good: save {energy_save:.1f}%, svc {svc_deg:.1f}% -> keep (nudge energy)'
    else:
        new_l=round(l,2); reason=f'stable band (svc {svc_deg:.1f}%) -> keep {e}/{l}'
    new_l=round(new_l,2); new_e=round(1.0-new_l,2)
    print(f'{new_e} {new_l}')
    print(f'---')
    print(f'Lambdas: {len(pairs_od)}')
    print(f'Energy vs ondemand (agent use Δ%): {en_od:+.1f}%  (positive=uses more)')
    print(f'Energy SAVE vs ondemand:           {energy_save:+.1f}%  (positive=good)')
    print(f'Service time vs ondemand:          {svc_deg:+.1f}%  (positive=slower)')
    print(f'Energy vs perf: {en_pf:+.1f}%   svc vs perf: {svc_pf:+.1f}%')
    print(f'Decision: {reason} -> next weights energy={new_e} latency={new_l}')

if __name__=='__main__':
    main()
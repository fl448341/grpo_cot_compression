#!/bin/bash

# used for testing whether kl_coef works as stated
# semi vibecoded with claude

set -u
cd ~/rl
OUT=phase2/results/kl_sign_test
mkdir -p "$OUT"
STEPS=15

clean_gpu () { pkill -9 -f main_ppo 2>/dev/null; pkill -9 -f 'ray::' 2>/dev/null; ray stop --force 2>/dev/null; sleep 5; }

run_beta () { # $1 = beta
    local B="$1"; local LOG="$OUT/train_beta${B}.log"
    echo "########## RUN lambda=0, beta(kl_loss_coef)=$B, $STEPS steps ##########"
    clean_gpu
    KL_COEF=$B LENGTH_LAMBDA=0.0 LENGTH_VARIANT=plain TOTAL_STEPS=$STEPS \
        SAVE_FREQ=999 TEST_FREQ=999 MAXRESP=512 GRPO_N=8 TEMP=0.7 LR=1e-5 \
        bash phase2/train_runpod.sh 2>&1 | tee "$LOG"
}

run_beta 0
run_beta 5

echo
echo "================= KL from base (actor/kl_loss) per step ================="
python3 - "$OUT/train_beta0.log" "$OUT/train_beta5.log" <<'PY'
import re, sys
def kl_seq(path):
    out=[]
    for line in open(path, errors='ignore'):
        m=re.search(r'actor/kl_loss:([0-9.eE+-]+)', line)
        if m:
            try: out.append(float(m.group(1)))
            except ValueError: pass
    return out
b0=kl_seq(sys.argv[1]); b5=kl_seq(sys.argv[2])
n=min(len(b0),len(b5))
print(f"{'step':>4} | {'beta=0 (natural)':>18} | {'beta=5':>10}")
for i in range(n):
    print(f"{i+1:>4} | {b0[i]:>18.4f} | {b5[i]:>10.4f}")
if n>=6:
    import statistics as st
    # mean over the second half (after warmup)
    h=n//2
    m0=st.mean(b0[h:n]); m5=st.mean(b5[h:n])
    print(f"\nmean KL (steps {h+1}-{n}):  beta=0 = {m0:.4f}   beta=5 = {m5:.4f}")
    print("="*70)
    if m5 < 0.6*m0:
        print(">>> CONCLUSION: KL(beta=5) CLEARLY LOWER -> the KL term is a PENALTY (anchor).")
        print(">>> Code is fine. Sign '-' stays. 'larger beta = more stable' is TRUE.")
    elif m5 > 1.6*m0:
        print(">>> CONCLUSION: KL(beta=5) CLEARLY HIGHER -> the KL term is a BONUS (pushes away from base).")
        print(">>> SIGN BUG. Flip '-' to '+' in dp_actor.py:270 and rerun the kl-scan.")
    else:
        print(">>> CONCLUSION: difference inconclusive. Raise beta further (e.g. 20) and repeat.")
else:
    print("\n(too few steps in the log - check that training started)")
PY

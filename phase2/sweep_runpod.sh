#!/bin/bash

# used for running training on each lambda with kl_coef > 0 (bad)
# also saving training checkpoints

set -u
cd ~/rl

KL_COEF=0.1
STEPS=15
SAVE_FREQ=3
BASE_MED=265                      
LAMBDAS="-0.05 0.0 0.05 0.1"
VARIANT=plain
LR=1e-5; TEMP=0.7; GRPO_N=8; MAXRESP=512; LENGTH_NORM=512

RESDIR=phase2/results
RUNS=$RESDIR/runs
mkdir -p "$RUNS"

clean_gpu () { pkill -9 -f main_ppo 2>/dev/null; pkill -9 -f 'ray::' 2>/dev/null; ray stop --force 2>/dev/null; sleep 5; }

echo "================== CLEANING OLD CHECKPOINTS =================="
rm -rf phase2/ckpts/*/actor/global_step_* 2>/dev/null
echo "Disk after cleanup:"; df -h ~ | tail -1

for LAM in $LAMBDAS; do
    TAG=lam${LAM}_kl${KL_COEF}
    EXPDIR=phase2/ckpts/grpo_lam${LAM}_${VARIANT}_kl${KL_COEF}/actor
    OUT=$RUNS/$TAG
    mkdir -p "$OUT"
    echo
    echo "########## RUN $TAG  (lambda=$LAM, kl_coef=$KL_COEF, steps=$STEPS) ##########"

    clean_gpu
    SAVE_FREQ=$SAVE_FREQ KL_COEF=$KL_COEF LENGTH_LAMBDA=$LAM LENGTH_VARIANT=$VARIANT \
        LENGTH_NORM=$LENGTH_NORM MAXRESP=$MAXRESP TOTAL_STEPS=$STEPS LR=$LR TEMP=$TEMP GRPO_N=$GRPO_N \
        bash phase2/train_runpod.sh 2>&1 | tee "$OUT/train.log"

    clean_gpu
    for ck in "$EXPDIR"/global_step_*; do
        [ -d "$ck" ] || continue
        s=$(basename "$ck" | grep -oE '[0-9]+')
        python -m phase2.eval_checkpoint --ckpt "$ck" --label ${TAG}_s$s \
            --out "$RESDIR" --max-new-tokens $MAXRESP --batch-size 32 \
            --lam $LAM --variant $VARIANT --lr $LR --steps $s --grpo-n $GRPO_N --temperature $TEMP
        mv "$RESDIR/records_${TAG}_s$s.json" "$OUT/" 2>/dev/null
    done

    python -m phase2.parse_trainlog "$OUT/train.log" "$OUT/trajectory.csv" 2>/dev/null || true

    THR=$(awk "BEGIN{print $BASE_MED*0.85}")
    BEST=$(awk -F, -v p="${TAG}_s" -v thr="$THR" '
        index($1,p)==1 {
            acc=$10+0; med=$12+0; st=$5;
            if (med<=thr && acc>ba) {ba=acc; bs=st}     # best accuracy among the compressed
            if (acc>aa)             {aa=acc; as=st}     # fallback: best accuracy overall
        }
        END { print (bs!=""?bs:as) }' "$RESDIR/phase2_eval.csv")
    echo "--- $TAG: best checkpoint = global_step_${BEST:-?} (median<=$THR, max acc); deleting the rest ---"
    for ck in "$EXPDIR"/global_step_*; do
        [ -d "$ck" ] || continue
        s=$(basename "$ck" | grep -oE '[0-9]+')
        [ -n "$BEST" ] && [ "$s" = "$BEST" ] || rm -rf "$ck"
    done
    echo "Disk after $TAG:"; df -h ~ | tail -1
done

mkdir -p runpod_sweep
cp "$RESDIR/phase2_eval.csv" runpod_sweep/ 2>/dev/null
rm -rf runpod_sweep/runs; cp -r "$RUNS" runpod_sweep/ 2>/dev/null

echo
echo "================== DONE =================="
echo "Eval results (all runs): $RESDIR/phase2_eval.csv"
echo "Per-triple (run log + trajectory + records): $RUNS/lam<L>_kl${KL_COEF}/"
echo "Kept model per lambda: phase2/ckpts/grpo_lam<L>_plain_kl${KL_COEF}/actor/  (best checkpoint)"
echo
cat "$RESDIR/phase2_eval.csv"

#!/bin/bash

# used for running training on each lambda with kl_coef < 0

set -u
cd ~/rl

KL_COEF=${KL_COEF:--0.01}
STEPS=${STEPS:-15}
SAVE_FREQ=${SAVE_FREQ:-3}
LAMBDAS=${LAMBDAS:-"0 0.01 0.05 0.1"}
GPUMEM=${GPUMEM:-0.1}
VARIANT=plain; LR=1e-5; TEMP=0.7; GRPO_N=8; MAXRESP=512; LENGTH_NORM=512

RESDIR=runpod_kneg
RUNS=$RESDIR/runs
mkdir -p "$RUNS"

clean_gpu () { pkill -9 -f main_ppo 2>/dev/null; pkill -9 -f 'ray::' 2>/dev/null; ray stop --force 2>/dev/null; sleep 5; }

echo "============ KNEG SWEEP  kl_coef=$KL_COEF (penalty), lambdas=[$LAMBDAS], steps=$STEPS ============"
for LAM in $LAMBDAS; do
    TAG=lam${LAM}_kl${KL_COEF}
    EXPDIR=phase2/ckpts/grpo_lam${LAM}_${VARIANT}_kl${KL_COEF}/actor
    OUT=$RUNS/$TAG; mkdir -p "$OUT"
    echo; echo "########## $TAG  (lambda=$LAM, kl_coef=$KL_COEF, steps=$STEPS) ##########"
    rm -rf "$EXPDIR"/global_step_* 2>/dev/null

    clean_gpu
    SAVE_FREQ=$SAVE_FREQ KL_COEF=$KL_COEF LENGTH_LAMBDA=$LAM LENGTH_VARIANT=$VARIANT \
        LENGTH_NORM=$LENGTH_NORM MAXRESP=$MAXRESP TOTAL_STEPS=$STEPS LR=$LR TEMP=$TEMP GRPO_N=$GRPO_N GPUMEM=$GPUMEM \
        bash phase2/train_runpod.sh 2>&1 | tee "$OUT/train.log"

    clean_gpu
    for ck in "$EXPDIR"/global_step_*; do
        [ -d "$ck" ] || continue
        s=$(basename "$ck" | grep -oE '[0-9]+')
        python -m phase2.eval_checkpoint --ckpt "$ck" --label ${TAG}_s$s --out "$RESDIR" \
            --max-new-tokens $MAXRESP --batch-size 32 --lam $LAM --variant $VARIANT \
            --lr $LR --steps $s --grpo-n $GRPO_N --temperature $TEMP
        mv "$RESDIR/records_${TAG}_s$s.json" "$OUT/" 2>/dev/null
    done
    python -m phase2.parse_trainlog "$OUT/train.log" "$OUT/trajectory.csv" 2>/dev/null || true

    rm -rf "$EXPDIR"/global_step_* 2>/dev/null
    echo "Disk after $TAG:"; df -h ~ 2>/dev/null | tail -1
done

echo; echo "============ SWEEP DONE ============"
echo "--- results ($RESDIR/phase2_eval.csv) ---"
cat "$RESDIR/phase2_eval.csv" 2>/dev/null

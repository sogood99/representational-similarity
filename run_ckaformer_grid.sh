#!/usr/bin/env bash


source /home/rdlvcs/.virtualenvs/ckaformer/bin/activate


GAMMAS=(
    "1e-5"
    "1e-4"
    "1e-3"
    "1e-2"
    "1e-1"
)

DEPTHS=(8 16 32 64)

TRAIN_MEANS=(
    "True"
    "False"
)


# Loop over the parameters
for GAMMA in "${GAMMAS[@]}"; do
    for DEPTH in "${DEPTHS[@]}"; do
        for TRAIN_MEAN in "${TRAIN_MEANS[@]}"; do
            python test_ckaformer.py \
                --gamma="$GAMMA" \
                --depth="$DEPTH" \
                --trainable_mean="$TRAIN_MEAN" || exit 1
        done
    done
done
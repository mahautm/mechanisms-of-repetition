#!/bin/bash
# Submit cycle 0 analysis for Pythia-70m (needed for alluvial plots)

for checkpoint in step1 step1000 step5000 step10000 step100000 steplatest; do
    SPECIFIC_OUTPUT="/home/mmahaut/projects/parrots/outputs_multihead_full/EleutherAI/pythia-70m/${checkpoint}/layer_4"
    mkdir -p ${SPECIFIC_OUTPUT}
    
    REV_FLAG=""
    if [ "${checkpoint}" != "steplatest" ]; then
        REV_FLAG="--revision=${checkpoint}"
    fi
    
    sbatch --job-name="cyc0_70m_${checkpoint}" \
        --output="${SPECIFIC_OUTPUT}/full_analysis_cyc0_ml32.out" \
        --error="${SPECIFIC_OUTPUT}/full_analysis_cyc0_ml32.err" \
        --mem=64G --partition=alien --qos=alien --gres=gpu:1 --time=2:00:00 \
        --wrap="source ~/.bashrc && conda activate parr && module load CUDA/12.2.0 && cd ~/projects/parrots/ && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && poetry run python /home/mmahaut/projects/parrots/parrots/aa_fortu/ckpt_pipeline_main.py --model-name=EleutherAI/pythia-70m --single-lens=4 --max-layer-idx=6 --lens-path=/home/mmahaut/projects/parrots/lenses_multihead/EleutherAI_pythia-70m/ --n-cycles=0 --max-length=32 --max-new-tokens=1000 --batch-size=8 --n-samples=1000 ${REV_FLAG}"
    
    sleep 0.3
done

echo "Submitted cycle 0 jobs for Pythia-70m layer 4"

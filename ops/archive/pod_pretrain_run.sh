#!/bin/bash
# THE CORPUS PASS ON A POD (2026-09-18, item 44): runs inside the pod. Expects in /workspace/pretrain: the body's code (body/,
# tools/), data/tok_char.json, the copy COPY.pt, the flags, the held-out and fact files, the page log; fetches the stories corpus;
# runs tools/pretrain_cortex.py on cuda; leaves OUT.pt and the log in /workspace/pretrain. Arguments: CHARS BATCH LR ROUNDS.
set -e
CHARS=${1:-100000000}; BATCH=${2:-64}; LR=${3:-1e-5}; ROUNDS=${4:-1}
cd /workspace/pretrain
pip install -q tokenizers 2>&1 | tail -1
if [ ! -s corpus/TinyStoriesV2-GPT4-train.txt ]; then
  mkdir -p corpus; curl -sL -o corpus/TinyStoriesV2-GPT4-train.txt "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
fi
ls -la corpus/ | tail -2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 tools/pretrain_cortex.py copy.pt --flags BASE_FLAGS.txt --corpus corpus/TinyStoriesV2-GPT4-train.txt --chars $CHARS --batch $BATCH --lr $LR --rounds $ROUNDS --device cuda --report-every 500 --save-as out.pt 2>&1 | grep --line-buffered -v "Warning\|^physiology" | tee pretrain_pod.log
echo "POD RUN DONE" >> pretrain_pod.log

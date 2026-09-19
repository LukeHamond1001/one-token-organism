#!/bin/zsh
# THE CORPUS PASS, ORCHESTRATED FROM THE MAC (2026-09-18, item 44): creates a GPU pod (REST, a preference list of GPUs, the EU-RO-1
# network volume at /workspace), waits for it, copies the code, the copy and the rulers' files up by scp, runs
# ops/pod_pretrain_run.sh there, copies out.pt and the log back, and removes the pod. The key in ~/.runpod_key.
#   ops/pod_pretrain.sh COPY.pt OUT.pt CHARS [BATCH] [LR]
set -u
COPY=$1; OUT=$2; CHARS=$3; BATCH=${4:-64}; LR=${5:-1e-5}
cd /Users/lukehamond/Projects/project; KEY=$(cat ~/.runpod_key)
BODY=$(cat <<JSON
{"name": "iga-pretrain", "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
 "gpuTypeIds": ["NVIDIA RTX A5000", "NVIDIA RTX A6000", "NVIDIA A40", "NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3090", "NVIDIA L4", "NVIDIA RTX 4000 Ada Generation", "NVIDIA A100 80GB PCIe", "NVIDIA L40S"],
 "gpuCount": 1, "cloudType": "SECURE", "containerDiskInGb": 40, "volumeInGb": 0,
 "networkVolumeId": "2o9gtwzkhd", "volumeMountPath": "/workspace", "ports": ["22/tcp"], "env": {}}
JSON
)
R=$(curl -s -X POST https://rest.runpod.io/v1/pods -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' -d "$BODY")
ID=$(echo "$R" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('id',''))" 2>/dev/null)
[ -z "$ID" ] && { echo "pod not created: $R"; exit 1; }
echo "$(date +%H:%M:%S) pod $ID created"
for i in $(seq 1 60); do
  P=$(curl -s https://rest.runpod.io/v1/pods/$ID -H "Authorization: Bearer $KEY")
  ST=$(echo "$P" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('desiredStatus',''), d.get('publicIp') or '', json.dumps(d.get('portMappings') or {}))" 2>/dev/null)
  echo "$ST" | grep -q "RUNNING" && echo "$ST" | grep -q '"22"' && break
  sleep 15
done
IP=$(echo "$ST" | awk '{print $2}'); PORT=$(echo "$ST" | python3 -c "import sys,json; s=sys.stdin.read().split(' ',2); print(json.loads(s[2]).get('22',''))")
[ -z "$IP" ] || [ -z "$PORT" ] && { echo "no ssh mapping: $ST"; exit 1; }
echo "$(date +%H:%M:%S) ssh root@$IP -p $PORT"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $PORT root@$IP"
for i in $(seq 1 20); do $SSH true 2>/dev/null && break; sleep 10; done
$SSH "mkdir -p /workspace/pretrain/data /workspace/pretrain/tools /workspace/pretrain/corpus"
scp -o StrictHostKeyChecking=no -P $PORT -r body tools/pretrain_cortex.py tools/heldout_stage4.txt tools/facts_stage5.txt ops/BASE_FLAGS.txt ops/pod_pretrain_run.sh root@$IP:/workspace/pretrain/ 2>&1 | tail -1
$SSH "cd /workspace/pretrain && mv pretrain_cortex.py heldout_stage4.txt facts_stage5.txt tools/ 2>/dev/null; true"
scp -o StrictHostKeyChecking=no -P $PORT data/tok_char.json data/watch2_caregiver.jsonl root@$IP:/workspace/pretrain/data/ 2>&1 | tail -1
scp -o StrictHostKeyChecking=no -P $PORT "$COPY" root@$IP:/workspace/pretrain/copy.pt 2>&1 | tail -1
echo "$(date +%H:%M:%S) uploaded; running"
$SSH "cd /workspace/pretrain && nohup bash pod_pretrain_run.sh $CHARS $BATCH $LR 1 > run.out 2>&1 &"
until $SSH "grep -q 'POD RUN DONE' /workspace/pretrain/pretrain_pod.log 2>/dev/null"; do sleep 120; $SSH "tail -1 /workspace/pretrain/pretrain_pod.log 2>/dev/null" | cut -c1-160; done
$SSH "cat /workspace/pretrain/pretrain_pod.log" > "${OUT%.pt}.log"
scp -o StrictHostKeyChecking=no -P $PORT root@$IP:/workspace/pretrain/out.pt "$OUT" 2>&1 | tail -1
echo "$(date +%H:%M:%S) result at $OUT; removing the pod"
curl -s -X DELETE https://rest.runpod.io/v1/pods/$ID -H "Authorization: Bearer $KEY" | head -c 200; echo

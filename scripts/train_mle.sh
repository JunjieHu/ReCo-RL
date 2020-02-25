#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

dir="${PWD}/downloads/"
objective="MLE"  # or "REINFORCE" or "MIXER"

output="$PWD/outputs/${objective}-lr0.0002-emb300-hid512-demo"
mkdir -p $output
model="${output}/model"
vocab="${output}/vocab.bin"
log="${output}/train.log"
save_decode_file="$output/decode-100"

CMD="python -u src/train.py \
    --vocab $vocab \
    --cuda \
    --save_model_to $model \
    --data_dir $dir \
    --batch_size 32 \
    --dropout 0.5 \
    --valid_interval 5000 \
    --log_interval 50 \
    --embed_size 300 \
    --hidden_size 512\
    --patience 10 \
    --lr 0.0002 \
    --objective ${objective} \
    --save_model_after 1 \
    --save_decode_file $save_decode_file \
    --log_file ${log}"
    
echo "$CMD > ${log}"
echo "$CMD" > ${log}
bash -c "$CMD" &>> ${log}

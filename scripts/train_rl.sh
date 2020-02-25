#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

# initialized from a MLE-trained model
init_dir="outputs/MLE-lr0.0002-emb300-hid512-demo/"
vocab="${init_dir}/vocab.bin"
load_model="${init_dir}/model.bin"

# data set path
data_dir="${PWD}/downloads/"

# path to Next-Sentence-Predictor model
bert_weight="$PWD/bert-weight/bert-base-uncased.tar.gz"
bert_vocab="$PWD/bert-weight/bert-base-uncased-vocab.txt"

# Tuning parameters
batch_size=16
lr=0.0001
lr_decay=0.8
objective="MIXER"  
rl_baseline='average'
rl_reward='expressiveness-coherence-relevance'
rl_weight=0.5
valid=5000
sample_size=5
# optim='sgd'
optim='adam'

# Save path & model
output="${out_dir}/${objective}-lr${lr}-reward${rl_reward}-demo"
mkdir -p $output
model="${output}/model"
log="${output}/train.rl.log"
decode_file="${output}/decode"

CMD="python -u vist/train_wx_coh.py \
    --vocab $vocab \
    --load_model_from $load_model \
    --save_model_to $model \
    --bert_weight_path $bert_weight \
    --bert_vocab_path $bert_vocab \
    --data_dir $data_dir \
    --batch_size $batch_size \
    --sample_size $sample_size \
    --log_interval 50 \
    --lr $lr \
    --rl_baseline $rl_baseline \
    --rl_weight $rl_weight \
    --lr_decay ${lr_decay} \
    --vocab $vocab \
    --decode_max_length 30 \
    --valid_interval $valid \
    --patience 10 \
    --save_model_after 1 \
    --save_decode_file $decode_file \
    --cuda \
    --rl_reward $rl_reward \
    --objective $objective \
    --log_file ${log} \
    --optim ${optim} "

echo "$CMD > ${log}"
echo "$CMD" > ${log}
bash -c "$CMD" &>> ${log}

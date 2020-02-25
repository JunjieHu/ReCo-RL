#!/bin/bash
export PYTHONPATH=.:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

dir="/home/junjieh/usr3/Research/AREL/VIST"
story_h5="$dir/story.h5"
story_line="$dir/story_line.json"
img_path="/home/junjieh/usr3/Research/AREL/VIST/"
new_vocab="$dir/spacy/vocab.spacy-lemma-lg.bin"

bert_weight="$PWD/bert-weight/bert-base-uncased.tar.gz"
bert_vocab="$PWD/bert-weight/bert-base-uncased-vocab.txt"

out_dir="$PWD/models/"
vocab="${out_dir}/vocab.bin"
load_model="${out_dir}/reco-mle-model.bin"

batch_size=16
lr=0.0001
lr_decay=0.8
rl_baseline='average'
# rl_reward='BLEU-coherence-dBLEU'
rl_reward='expressiveness-coherence-relevance'
rl_weight=0.5
valid=5000
sample_size=5
# objective='MLE'
# objective='REINFORCE'
objective='MIXER'
# optim='sgd'
optim='adam'

output="${out_dir}/acl/rl_avg_bt${batch_size}_lr${lr}_BL${rl_baseline}_valid${valid}_lrdecay${lr_decay}_word_reward${rl_reward}_rlweight${rl_weight}_obj${objective}_optim${optim}_sample${sample_size}"
mkdir -p $output
model="${output}/model"
log="${output}/train.rl.log"
decode_file="${output}/decode-len30"

CMD="python -u vist/train_wx_coh.py \
    --vocab $vocab \
    --load_model_from $load_model \
    --new_vocab $new_vocab \
    --save_model_to $model \
    --bert_weight_path $bert_weight \
    --bert_vocab_path $bert_vocab \
    --story_h5 $story_h5 \
    --story_line $story_line \
    --wx_data_dir $dir \
    --img_path $img_path \
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
    --rl_baseline $rl_baseline \
    --arch feudal \
    --rl_reward $rl_reward \
    --valid_metric Bleu_4 \
    --objective $objective \
    --log_file ${log} \
    --optim ${optim} "

echo "$CMD > ${log}"
echo "$CMD" > ${log}
bash -c "$CMD" &>> ${log}

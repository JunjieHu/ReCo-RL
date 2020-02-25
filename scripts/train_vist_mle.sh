#!/bin/bash
export PYTHONPATH=.:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

# data_dir="../data/visual-storytelling/processed/data/"
# dataset_file="${data_dir}/dataset.p"
# ref_file="${data_dir}/dataset_refs.p"
# img_feats_train_file="${data_dir}/img_feats_train.mat"
# img_feats_dev_file="${data_dir}/img_feats_val.mat"
# img_feats_test_file="${data_dir}/img_feats_test.mat"

# dir="/home/junjieh/usr3/Research/AREL/VIST/debug"
dir="/home/junjieh/usr3/Research/AREL/VIST"
story_h5="$dir/story.h5"
story_line="$dir/story_line.json"
img_path="/home/junjieh/usr3/Research/AREL/VIST/"


output="$PWD/outputs/baseline-lr0.0002-emb300-hid512-demo"
mkdir -p $output
model="${output}/model"
vocab="${output}/vocab.bin"
log="${output}/train.log"
save_decode_file="$output/decode-100"

CMD="python -u src/train_wx_coh.py \
    --vocab $vocab \
    --cuda \
    --save_model_to $model \
    --story_h5 $story_h5 \
    --story_line $story_line \
    --wx_data_dir $dir \
    --img_path $img_path \
    --batch_size 32 \
    --dropout 0.5 \
    --valid_interval 5000 \
    --log_interval 50 \
    --embed_size 300 \
    --hidden_size 512\
    --patience 10 \
    --lr 0.0002 \
    --objective MLE \
    --save_model_after 1 \
    --save_decode_file $save_decode_file \
    --arch vist_mimg_enc \
    --log_file ${log}"
    # --bottomup_combine \
echo "$CMD > ${log}"
echo "$CMD" > ${log}
bash -c "$CMD" &>> ${log}

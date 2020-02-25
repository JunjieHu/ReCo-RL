#!/bin/bash
export PYTHONPATH=.:$PYTHONPATH
# export PYTHONPATH=./pycocoevalcap:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

ROOT="/home/jjhu/usr3/"

dir="${ROOT}/Research/AREL/VIST"
story_h5="$dir/story.h5"
story_line="$dir/story_line.json"
img_path="${ROOT}/Research/AREL/VIST/"

# output="/home/junjieh/usr3/Research/seq2seq-backup/output/vist/baseline-mimg-dropout0.5-notie"
# output="/home/junjieh/usr3/Research/seq2seq/output/vist/wx/baseline-lr0.0002-emb300-hid512-nobottomup/"
# output="/home/junjieh/usr3/Research/seq2seq/output/vist/wx/baseline-lr0.0002-emb300-hid512/"
#output="${ROOT}/Research/seq2seq/aws1/seq2seq/output/vist/wx/baseline-lr0.0002-emb300-hid512-notieweight/"
# output="/home/junjieh/usr3/Research/seq2seq/aws3/acl-coh/rl_avg_bt16_lr0.00005_BLaverage_valid1000_lrdecay0.8_word_rewardentBLEU-coherence-dBLEU_rlweight0.5_objMIXER_optimadam_sample5_coh1_div1_ent1_entWeight0.25-0.25-0.25-0.25_entBeta10/"
# mkdir -p $output

output="outputs/rl/"
model="${output}/model.bin"
vocab="${output}/vocab.bin"
#vocab="models/vocab.bin"
#model="models/reco-mle-model.bin"
#output="outputs/"
an=2
dec_len=10
#mkdir -p $output/reco/
log="${output}/test.beam5.wx-single-beam-debug-avoid${an}-debug-dlen${dec_len}.log"
decode_file="${output}/decode-test-len30-single-beam5-wx-bug-avoid${an}-gram-debug-dlen${dec_len}"

CMD="python -u src/test.py \
    --vocab $vocab \
    --avoid_ngram $an \
    --load_model_from $model \
    --story_h5 $story_h5 \
    --story_line $story_line \
    --wx_data_dir $dir \
    --img_path $img_path \
    --batch_size 128 \
    --decode_max_length 30 \
    --cuda \
    --arch feudal \
    --decode_type beam \
    --beam_size 5 \
    --decode_len_constraint $dec_len \
    --save_decode_file ${decode_file}" #\
    # --save_nbest_format cdec "

echo "$CMD > ${log}"
echo "$CMD" > ${log}
bash -c "$CMD" &>> ${log}

# # source activate py27
# CMD="python vist/scorer.py $ref_file $decode_file "
# echo "$CMD >> ${log}"
# echo "CMD" >> ${log}
# bash -c "$CMD >> ${log}"

python src/scorer.py ../AREL/VIST/test_dataset.p $decode_file >> ${log} 

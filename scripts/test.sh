#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

# REPO: path to the repository
REPO=${PWD}
data_dir=${REPO}/downloads/

output="${REPO}/outputs/rl/"
model="${output}/model.bin"
vocab="${output}/vocab.bin"
dec_len=30
log="${output}/decode-beam5-len${dec_len}.log"
decode_file="${output}/decode-beam5-len${dec_len}.tsv"

CMD="python -u $REPO/src/test.py \
    --vocab $vocab \
    --load_model_from $model \
    --data_dir $data_dir \
    --batch_size 128 \
    --cuda \
    --decode_type beam \
    --beam_size 5 \
    --decode_len_constraint $dec_len \
    --save_decode_file ${decode_file}" 

#echo "$CMD > ${log}"
#echo "$CMD" > ${log}
#bash -c "$CMD" &>> ${log}

# # source activate py27
# CMD="python vist/scorer.py $ref_file $decode_file "
# echo "$CMD >> ${log}"
# echo "CMD" >> ${log}
# bash -c "$CMD >> ${log}"

python $REPO/src/scorer.py $data_dir/test_dataset.p $decode_file >> ${log} 

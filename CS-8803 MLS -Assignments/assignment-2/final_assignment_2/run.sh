#!/bin/bash

# constructing poisoned data

python construct_poisoned_data.py --input_dir 'SST2' \
        --output_dir 'SST2_poisoned' --poisoned_ratio 0.01 \
        --target_label 1 --trigger_word 'cc'

# EP attacking

python ep_train.py --clean_model_path 'SST2_clean_model' --epochs 3 \
        --data_dir 'SST2_poisoned' \
        --save_model_path 'SST2_EP_model' --batch_size 32 \
        --lr 5e-2 --trigger_word 'cc'

# calculating clean acc. and ASR

python test_asr.py --model_path 'SST2_clean_model' \
        --data_dir 'SST2' \
        --batch_size 32  \
        --trigger_word 'cc' --target_label 1

python test_asr.py --model_path 'SST2_EP_model' \
        --data_dir 'SST2' \
        --batch_size 32  \
        --trigger_word 'cc' --target_label 1

# running defense

python defense.py
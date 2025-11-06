#!/bin/bash

# This script is used for stage 1 training of Orion-BiX

# ----------------------------------
# Generate prior datasets on the fly
# ----------------------------------

torchrun --standalone --nproc_per_node=1 /path/to/orion_bix/train/run.py \
            --meta_learning True \
            --support_size 64 \
            --query_size 64 \
            --k_neighbors 128 \
            --similarity_metric cosine \
            --feature_normalization True \
            --num_episodes_per_dataset 10 \
            --diversity_factor 0.3 \
            --min_dataset_size 1024 \
            --episode_method random \
            --wandb_log True \
            --wandb_project Orion-BiX \
            --wandb_name Stage1 \
            --wandb_dir /my/wandb/dir \
            --wandb_mode online \
            --device cuda \
            --dtype float32 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 25000 \
            --batch_size 2048 \
            --micro_batch_size 4 \
            --lr 1e-4 \
            --scheduler cosine_warmup \
            --warmup_proportion 0.02 \
            --gradient_clipping 1.0 \
            --prior_type mix_scm \
            --prior_device cpu \
            --batch_size_per_gp 4 \
            --min_features 2 \
            --max_features 100 \
            --max_classes 10 \
            --max_seq_len 1024 \
            --min_train_size 0.1 \
            --max_train_size 0.9 \
            --embed_dim 128 \
            --col_num_blocks 3 \
            --col_nhead 4 \
            --col_num_inds 128 \
            --col_attention_type linear \
            --col_feature_map elu \
            --row_num_blocks 3 \
            --row_nhead 8 \
            --row_num_cls 4 \
            --row_rope_base 100000 \
            --row_attention_type bi_axial \
            --row_feature_map elu \
            --icl_num_blocks 12 \
            --icl_nhead 4 \
            --ff_factor 2 \
            --icl_attention_type standard \
            --icl_feature_map elu \
            --norm_first True \
            --checkpoint_dir /my/stage1/checkpoint/dir \
            --save_temp_every 50 \
            --save_perm_every 1000



# Loading prior data from disk and training
torchrun --standalone --nproc_per_node=1 /path/to/orion_bix/train/run.py \
            --meta_learning True \
            --support_size 64 \
            --query_size 64 \
            --k_neighbors 128 \
            --similarity_metric cosine \
            --feature_normalization True \
            --num_episodes_per_dataset 10 \
            --diversity_factor 0.3 \
            --min_dataset_size 1024 \
            --episode_method random \
            --wandb_log True \
            --wandb_project Orion-BiX \
            --wandb_name Stage1 \
            --wandb_dir /my/wandb/dir \
            --wandb_mode online \
            --device cuda \
            --dtype float32 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 25000 \
            --batch_size 2048 \
            --micro_batch_size 4 \
            --lr 1e-4 \
            --scheduler cosine_warmup \
            --warmup_proportion 0.02 \
            --gradient_clipping 1.0 \
            --prior_dir /my/stage1/prior/dir \
            --load_prior_start 0 \
            --delete_after_load False \
            --prior_device cpu \
            --embed_dim 128 \
            --col_num_blocks 3 \
            --col_nhead 4 \
            --col_num_inds 128 \
            --col_attention_type linear \
            --col_feature_map elu \
            --row_num_blocks 3 \
            --row_nhead 8 \
            --row_num_cls 4 \
            --row_rope_base 100000 \
            --row_attention_type bi_axial \
            --row_feature_map elu \
            --icl_num_blocks 12 \
            --icl_nhead 4 \
            --ff_factor 2 \
            --icl_attention_type standard \
            --icl_feature_map elu \
            --norm_first True \
            --checkpoint_dir /my/stage1/checkpoint/dir \
            --save_temp_every 50 \
            --save_perm_every 100\

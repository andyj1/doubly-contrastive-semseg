#!/usr/bin/env bash

#env setting
conda env create --file environment.yaml
cd network/deform_conv && bash build.sh


#############################################################################################################3
## sceneflow
# for training
python main.py --gpu_id 0 --dataset sceneflow --checkname disp_only_resnet18_train_scene_flow_3type \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 350 \
--with_refine --refinement_type hourglass --batch_size 8 --val_batch_size 8 --train_disparity

# for test
python main.py --gpu_id 0 --dataset sceneflow --checkname disp_only_resnet18_train_scene_flow_3type_test \
--refinement_type hourglass --val_batch_size 1 --train_disparity --with_refine \
--resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth --test_only
# (epe:0.723, d1: 0.0348, >1px:0.0883, mean time:0.067s)

# ---------------------------------------------
## cityscapes
# for training
python main.py --gpu_id 0 --dataset cityscapes --checkname resnet18_train_cityscapes_transfer \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4  --epochs 400 \
--batch_size 4 --val_batch_size 4 --train_disparity --train_semantic --with_refine \
--refinement_type ours --resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth \
--transfer_disparity

# for test (full-size image)
python main.py --gpu_id 0 --dataset cityscapes --checkname resnet18_train_cityscapes_transfer_test \
--val_batch_size 1 \
--train_disparity --train_semantic --with_refine --refinement_type ours \
--resume run/cityscapes/resnet18_train_cityscapes_transfer/experiment_0/score_best_checkpoint.pth --test_only
# (mIoU:76.4%, epe:0.76, d1: 2.55%, mean time:0.054s(18.5fps))

  #(half-size image)
python main.py --gpu_id 0 --dataset cityscapes --checkname resnet18_train_cityscapes_transfer_test \
--val_batch_size 1 \
--train_disparity --train_semantic --with_refine --refinement_type ours \
--resume run/cityscapes/resnet18_train_cityscapes_transfer/experiment_0/score_best_checkpoint.pth --test_only \
--val_img_height 512 --val_img_width 1024
# (mIoU:0.707, epe:1.35, d1: 0.0776, >1px:0.152, mean time:0.033s(29.4fps))

# ---------------------------------------------
## cityscapes + lost_and_found
# for training
# 1. without transfer
python main.py --gpu_id 0 --dataset city_lost --checkname resnet18_train_citylost_eps_1e-1_without_transfer \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--train_semantic --train_disparity --with_refine --refinement_type ours --batch_size 4 --val_batch_size 4 \
--epsilon 1e-1

# 2. with sceneflow transfer methods
python main.py --gpu_id 0 --dataset city_lost --model ournet_resnet18 --checkname resnet18_train_citylost_eps_1e-1 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type ours --batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity --resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth \
--transfer_disparity --epsilon 1e-1

# for test (full-size image)
python main.py --gpu_id 0 --dataset city_lost --checkname resnet18_train_citylost_eps_1e-1_test \
--with_refine  --refinement_type ours --val_batch_size 1 --train_semantic --train_disparity \
--resume run/city_lost/resnet18_train_citylost_eps_1e-1/experiment_0/score_best_checkpoint.pth --test_only
# (mIoU:74.1%, epe:1.20, d1: 4.73%, mean time:0.052s(18.5fps))
# to save validation_results : add --save_val_results

  #(half-size image)
python main.py --gpu_id 0 --dataset city_lost --checkname resnet18_train_citylost_eps_1e-1_test \
--with_refine  --refinement_type ours --val_batch_size 1 --train_semantic --train_disparity \
--resume run/city_lost/resnet18_train_citylost_eps_1e-1/experiment_0/score_best_checkpoint.pth --test_only \
--val_img_height 512 --val_img_width 1024
# 0.042s


# ---------------------------------------------
# KITTI_2015
# for training
# train kitti_2015 + kitti_2012 from sceneflow transfered results with only disparity modules
python main.py --gpu_id 0 --dataset kitti_mix \
--checkname train_kitti_mix_with_sceneflow_transfer --optimizer_policy ADAM --lr 4e-4 \
--weight_decay 1e-4 --last_lr 1e-6 --epochs 600 --with_refine --refinement_type ours \
--batch_size 5 --val_batch_size 5 --train_disparity --train_semantic \
--resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth \
--without_balancing

# train kitti_2015 only with transfer above results
python main.py --gpu_id 0 \
--dataset kitti_2015 \
--checkname train_kitti_2015_transfer_from_kitti_mix_sceneflow_final \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --last_lr 1e-6 \
--epochs 600 \
--with_refine \
--refinement_type ours \
--batch_size 10 --val_batch_size 10 \
--train_disparity --train_semantic \
--resume run/kitti_mix/train_kitti_mix_with_sceneflow_transfer/experiment_0/epe_best_checkpoint.pth \
--epsilon 1e-1 --sem_weight 3 --disp_weight 10

# test
python main.py --gpu_id 0 \
--dataset kitti_2015 \
--checkname train_kitti_2015_transfer_from_kitti_mix_sceneflow_final_test \
--with_refine \
--refinement_type ours \
--val_batch_size 1 \
--train_disparity --train_semantic \
--resume run/kitti_2015/train_kitti_2015_transfer_from_kitti_mix_sceneflow_final/experiment_0/epe_best_checkpoint.pth \
--test_only





















python main.py --gpu_id 0 \
--dataset kitti_2015 \
--checkname train_kitti_2015_resnet18_12 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --last_lr 1e-6 \
--epochs 1000 \
--with_refine \
--refinement_type ours \
--batch_size 8 --val_batch_size 8 \
--train_disparity --train_semantic \
--resume run/cityscapes/resnet18_train_cityscapes_transfer/experiment_0/epe_best_checkpoint.pth \
--load_pseudo_gt


# for test accuracy with validation sets
python main.py --gpu_id 0 --dataset kitti_2015 \
--checkname train_kitti_2015_resnet18_12_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --last_lr 1e-6 --epochs 1000 \
--feature_type ganet --feature_pyramid --refinement_type ours \
--batch_size 8 --val_batch_size 1 --train_disparity --train_semantic --with_refine \
--resume run/kitti_2015/train_kitti_2015_resnet18_12/experiment_0/epe_best_checkpoint.pth \
--test_only --save_val_results


# Inference on KITTI 2015 test set for submission
python inference.py --gpu_id 0 \
--data_root /root/dataset \
--dataset kitti_2015 \
--with_refine \
--refinement_type ours \
--val_batch_size 1 \
--val_img_height 384 --val_img_width 1280 \
--train_disparity --train_semantic \
--resume run/kitti_2015/train_kitti_2015_resnet18_12/experiment_0/epe_best_checkpoint.pth \
--output_dir output/kitti15_test



# 07.20
python main.py --gpu_id 1 \
--dataset kitti_2015 \
--checkname train_kitti_2015_transfer_from_sceneflow \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --last_lr 1e-6 \
--epochs 600 \
--with_refine \
--refinement_type ours \
--batch_size 10 --val_batch_size 10 \
--train_disparity --train_semantic \
--resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth \
--epsilon 1e-1 --sem_weight 7 --disp_weight 10
new: 1.9D1, 61.3mIoU (disp + residual*scale ) // origin(deconv bias=True): 1.78d1, 62.59mIoU (exp2)

07.21
python main.py --gpu_id 1 \
--dataset kitti_2015 \
--checkname train_kitti_2015_transfer_from_sceneflow \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --last_lr 1e-6 \
--epochs 600 \
--with_refine \
--refinement_type ours \
--batch_size 10 --val_batch_size 10 \
--train_disparity --train_semantic \
--resume run/kitti_2015/train_kitti_2015_transfer_from_sceneflow/experiment_2/epe_best_checkpoint.pth \
--epsilon 1e-1 --sem_weight 3 --disp_weight 10
 // origin(deconv bias=True): 0.992d1 (exp3) : sceneflow --> sceneflow again




python main.py --gpu_id 0 \
--dataset kitti_2015 \
--checkname train_kitti_2015_transfer_from_cityscapes \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --last_lr 1e-6 \
--epochs 600 \
--with_refine \
--refinement_type ours \
--batch_size 10 --val_batch_size 10 \
--train_disparity --train_semantic \
--resume run/cityscapes/resnet18_train_cityscapes_transfer/experiment_0/score_best_checkpoint.pth \
--epsilon 1e-1 --sem_weight 0.5 --disp_weight 10
new: 2.08d1, 67.7mIoU  (disp + residual*scale ) // origin(deconv bias=True): 2.03d1, 67.68,IoU

#------------------------------------------------------------------------
# KITTI_MIX training

python main.py --gpu_id 0 --dataset kitti_mix \
--checkname train_kitti_mix_with_sceneflow_transfer --optimizer_policy ADAM --lr 4e-4 \
--weight_decay 1e-4 --last_lr 1e-6 --epochs 600 --with_refine --refinement_type ours \
--batch_size 5 --val_batch_size 5 --train_disparity --train_semantic \
--resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth \
--without_balancing
2.1d1 (exp 0)  / retrain with kitti_mix_all_val (d1:0.945)

python main.py --gpu_id 2 --dataset kitti_mix \
--checkname train_kitti_mix_with_cityscapes_transfer --optimizer_policy ADAM --lr 4e-4 \
--weight_decay 1e-4 --last_lr 1e-6 --epochs 600 --with_refine --refinement_type ours \
--batch_size 10 --val_batch_size 10 --train_disparity --train_semantic \
--resume run/cityscapes/resnet18_train_cityscapes_transfer/experiment_0/score_best_checkpoint.pth \
--without_balancing
2.07d1



python main.py --gpu_id 3 \
--dataset kitti_2015 \
--checkname train_kitti_2015_transfer_from_kitti_mix_cityscapes \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --last_lr 1e-6 \
--epochs 600 \
--with_refine \
--refinement_type ours \
--batch_size 10 --val_batch_size 10 \
--train_disparity --train_semantic \
--resume run/kitti_mix/train_kitti_mix_with_cityscapes_transfer/experiment_0/epe_best_checkpoint.pth \
--epsilon 1e-1 --sem_weight 3 --disp_weight 10
d1:1.84 mIoU:67.49 (test quality not good)


python main.py --gpu_id 3 \
--dataset kitti_2015 \
--checkname train_kitti_2015_transfer_from_kitti_mix_sceneflow \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --last_lr 1e-6 \
--epochs 600 \
--with_refine \
--refinement_type ours \
--batch_size 10 --val_batch_size 10 \
--train_disparity --train_semantic \
--resume run/kitti_mix/train_kitti_mix_with_sceneflow_transfer/experiment_0/epe_best_checkpoint.pth \
--epsilon 1e-1 --sem_weight 3 --disp_weight 10
0.86D1, 82.70mIoU





python inference_color.py --gpu_id 3 \
--data_root /root/dataset \
--dataset kitti_2015 \
--with_refine \
--refinement_type ours \
--val_batch_size 1 \
--val_img_height 384 --val_img_width 1280 \
--train_disparity --train_semantic \
--resume run/kitti_mix/train_kitti_mix_with_cityscapes_transfer/experiment_0/epe_best_checkpoint.pth \
--output_dir output/train_kitti_mix_with_cityscapes_transfer


python inference_color.py --gpu_id 3 \
--data_root /root/dataset \
--dataset kitti_2015 \
--with_refine \
--refinement_type ours \
--val_batch_size 1 \
--val_img_height 384 --val_img_width 1280 \
--train_disparity --train_semantic \
--resume run/kitti_2015/train_kitti_2015_transfer_from_cityscapes/experiment_1/epe_best_checkpoint.pth \
--output_dir output/train_kitti_2015_transfer_from_cityscapes


python main.py --gpu_id 1 \
--dataset kitti_2015 \
--checkname train_kitti_2015_transfer_from_kitti_mix_sceneflow_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --last_lr 1e-6 \
--epochs 600 \
--with_refine \
--refinement_type ours \
--batch_size 10 --val_batch_size 1 \
--train_disparity --train_semantic \
--resume run/kitti_2015/train_kitti_2015_transfer_from_kitti_mix_sceneflow/experiment_0/score_best_checkpoint.pth \
--epsilon 1e-1 --sem_weight 3 --disp_weight 10 --test_only \
--save_val_results


python inference_color.py --gpu_id 3 \
--data_root /root/dataset \
--dataset kitti_2015 \
--with_refine \
--refinement_type ours \
--val_batch_size 1 \
--val_img_height 384 --val_img_width 1280 \
--train_disparity --train_semantic \
--resume run/kitti_2015/train_kitti_2015_transfer_from_kitti_mix_sceneflow_final/experiment_0/epe_best_checkpoint.pth \
--output_dir output/train_kitti_2015_transfer_from_kitti_mix_sceneflow_final

python inference_color.py --gpu_id 1 \
--data_root /root/dataset \
--dataset kitti_2015 \
--with_refine \
--refinement_type ours \
--val_batch_size 1 \
--val_img_height 384 --val_img_width 1280 \
--train_disparity --train_semantic \
--resume run/kitti_2015/train_kitti_2015_transfer_from_kitti_mix_sceneflow_all_val/experiment_0/epe_best_checkpoint.pth \
--output_dir output/train_kitti_2015_transfer_from_kitti_mix_sceneflow_all_val









## epsilon ablation study [1e-2, 2e-2, 5e-2, 1e-1]
python main.py --gpu_id 0 --dataset city_lost --model ournet_resnet18 --checkname resnet18_train_citylost_eps_1e-2 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type ours --batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity --resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth \
--transfer_disparity --epsilon 1e-2

python main.py --gpu_id 1 --dataset city_lost --model ournet_resnet18 --checkname resnet18_train_citylost_eps_2e-2 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type ours --batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity --resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth \
--transfer_disparity --epsilon 2e-2

python main.py --gpu_id 2 --dataset city_lost --model ournet_resnet18 --checkname resnet18_train_citylost_eps_5e-2 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type ours --batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity --resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth \
--transfer_disparity --epsilon 5e-2

python main.py --gpu_id 3 --dataset city_lost --model ournet_resnet18 --checkname resnet18_train_citylost_eps_1e-1 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type ours --batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity --resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth \
--transfer_disparity --epsilon 1e-1

python main.py --gpu_id 0 --dataset city_lost --model ournet_resnet18 --checkname resnet18_train_citylost_eps_0 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type ours --batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity --resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth \
--transfer_disparity --epsilon 0



# test
python main.py --gpu_id 3 --dataset city_lost --model ournet_resnet18 --checkname resnet18_train_citylost_eps_1e-1_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type ours --batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity --resume run/city_lost/resnet18_train_citylost_eps_1e-1/experiment_0/score_best_checkpoint.pth \
--epsilon 1e-1 --test_only





## retraining cityscapes
# with transfer disparity
python main.py --gpu_id 2 --dataset cityscapes --checkname resnet18_train_cityscapes_transfer \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4  --epochs 400 \
--batch_size 4 --val_batch_size 4 --train_disparity --train_semantic --with_refine \
--refinement_type ours --resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth \
--transfer_disparity --epsilon 1e-1

# without transfer disparity
python main.py --gpu_id 1 --dataset cityscapes --checkname resnet18_train_cityscapes_without_transfer \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4  --epochs 400 \
--batch_size 4 --val_batch_size 4 --train_disparity --train_semantic --with_refine \
--refinement_type ours --epsilon 1e-1

# with transfer disparity epsilon 0
python main.py --gpu_id 3 --dataset cityscapes --checkname resnet18_train_cityscapes_transfer_epsilon_0 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4  --epochs 400 \
--batch_size 4 --val_batch_size 4 --train_disparity --train_semantic --with_refine \
--refinement_type ours --resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth \
--transfer_disparity --epsilon 0
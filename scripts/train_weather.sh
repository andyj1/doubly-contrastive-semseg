# 1. train with automatic mixed precisions (amp) : add --amp command
python main.py --gpu_id 0 --dataset acdc --model resnet18 --checkname resnet18_train_acdc --optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 --batch_size 8 --val_batch_size 8 --train_semantic --epsilon 1e-1

w/o amp : 209.6s for training one epochs
w amp:



python main.py --gpu_id 0 --dataset acdc_city \
--model resnet34 --checkname resnet34_train_acdc_cityfull \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 --batch_size 8 --val_batch_size 8 \
--train_semantic --epsilon 1e-1 --acdc_cityfull


# gamma correction(0.4) to night image
python main.py --gpu_id 3 --dataset acdc_city \
--model resnet34 --checkname resnet34_train_acdc_cityfull_add_gamma_corr \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 --batch_size 8 --val_batch_size 8 \
--train_semantic --epsilon 1e-1 --acdc_cityfull \
--use_gamma_correction


python main.py --gpu_id 2 --dataset acdc_city \
--model resnet18 --checkname resnet18_train_acdc_cityfull \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 --batch_size 8 --val_batch_size 8 \
--train_semantic --epsilon 1e-1 --acdc_cityfull
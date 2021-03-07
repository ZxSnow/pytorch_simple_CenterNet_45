python train.py --log_name res2dcn50 \
                --dataset pascal \
                --arch resdcn_50 \
                --img_size 512 \
                --lr 1.25e-4 \
                --lr_step 45,60 \
                --batch_size 10 \
                --num_epochs 70 \
                --num_workers 10 \
                --data_dir /home/zx/datasets/pascal/
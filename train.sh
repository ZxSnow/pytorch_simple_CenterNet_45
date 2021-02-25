python train.py --log_name pascal_resdcn18 \
                --dataset pascal \
                --arch resdcn_18 \
                --img_size 512 \
                --lr 1.25e-4 \
                --lr_step 45,60 \
                --batch_size 32 \
                --num_epochs 70 \
                --num_workers 10 \
                --data_dir /home/zx/datasets/pascal/
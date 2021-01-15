mkdir -pv checkpoint/GD1-2
CUDA_VISIBLE_DEVICES=1 nohup python grafting.py --arch resnet32 --num_classes 10  --lr 0.1  --i 1 --num 2 --s checkpoint/GD1-2 --teacher_arch resnet56 --teacher_dir checkpoint/baseline/best_1.t7 >checkpoint/GD1-2/1.log &
CUDA_VISIBLE_DEVICES=2 nohup python grafting.py --arch resnet32 --num_classes 10  --lr 0.05 --i 2 --num 2 --s checkpoint/GD1-2 --teacher_arch resnet56 --teacher_dir checkpoint/baseline/best_1.t7 >checkpoint/GD1-2/2.log &
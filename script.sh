export CUDA_VISIBLE_DEVICES=0,4,5,6,7
torchrun --nproc_per_node=5 main.py
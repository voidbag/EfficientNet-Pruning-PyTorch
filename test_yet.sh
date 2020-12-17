CUDA_VISIBLE_DEVICES="" python3 ./gen_origin_from_pretrained_for_test.py --efficientnet-kind efficientnet-b1 --efficientnet-repo yet --save-path ./b1_yet.pth 

CUDA_VISIBLE_DEVICES="" python3 ./prune_efficientnet.py --efficientnet-kind efficientnet-b1 --efficientnet-repo yet --save-path ./b1_yet_pruned.pth --load-path ./b1_yet.pth --prune-ratio 0.9

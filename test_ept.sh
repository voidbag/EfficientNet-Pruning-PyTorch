CUDA_VISIBLE_DEVICES="" python3 ./gen_origin_from_pretrained_for_test.py --efficientnet-kind efficientnet-b1 --efficientnet-repo ept --save-path ./b1_ept.pth 

CUDA_VISIBLE_DEVICES="" python3 ./prune_efficientnet.py --efficientnet-kind efficientnet-b1 --efficientnet-repo ept --save-path ./b1_ept_pruned.pth --load-path ./b1_ept.pth --prune-ratio 0.9

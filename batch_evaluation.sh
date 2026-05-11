### 1. refine_step_prompt——scannet
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/refine_step_prompt_scannet_test.jsonl\
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/refine_step_prompt/scannet/step_mix_refine0227_0015000 \
#   --model_path /mnt/inspurfs/efm_t/linjingli/model/bagel_merge/step_mix_refine0227_0015000 \
#   --add_context_role_text


### 2. refine_step_prompt——dl3dv
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/refine_step_prompt_dl3dv_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/refine_step_prompt/dl3dv/step_mix_refine0227_0015000 \
#   --model_path /mnt/inspurfs/efm_t/linjingli/model/bagel_merge/step_mix_refine0227_0015000 \
#   --add_context_role_text


### 3. refine_step_prompt——matterport3d
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/refine_step_prompt_matterport3d_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/refine_step_prompt/matterport3d/step_mix_refine0227_0015000 \
#   --model_path /mnt/inspurfs/efm_t/linjingli/model/bagel_merge/step_mix_refine0227_0015000 \
#   --add_context_role_text



## 4. step_prompt——scannet
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/step_prompt_scannet_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/step_prompt/scannet/step_mix0227_0015000 \
#   --model_path /mnt/inspurfs/efm_t/linjingli/model/bagel_merge/step_mix0227_0015000 \
#   --add_context_role_text


### 5. step_prompt——dl3dv
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/step_prompt_dl3dv_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/step_prompt/dl3dv/step_mix0227_0015000 \
#   --model_path /mnt/inspurfs/efm_t/linjingli/model/bagel_merge/step_mix0227_0015000 \
#   --add_context_role_text


### 6. step_prompt——matterport3d
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/step_prompt_matterport3d_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/step_prompt/matterport3d/step_mix0227_0015000 \
#   --model_path /mnt/inspurfs/efm_t/linjingli/model/bagel_merge/step_mix0227_0015000 \
#   --add_context_role_text


### 7. onlyone_refine_step_prompt——scannet
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/onlyone_refine_step_prompt_scannet_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/onlyone_refine_step_prompt/scannet/step_mix_refine_onlyone0227_0015000 \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/models/ft_weights_old/step_mix_refine_onlyone0227_0015000 \
#   --add_context_role_text


### 8. onlyone_refine_step_prompt——dl3dv
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/onlyone_refine_step_prompt_dl3dv_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/onlyone_refine_step_prompt/dl3dv/step_mix_refine_onlyone0227_0015000 \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/models/ft_weights_old/step_mix_refine_onlyone0227_0015000 \
#   --add_context_role_text


### 9. onlyone_refine_step_prompt——matterport3d
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/onlyone_refine_step_prompt_matterport3d_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/onlyone_refine_step_prompt/matterport3d/step_mix_refine_onlyone0227_0015000 \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/models/ft_weights_old/step_mix_refine_onlyone0227_0015000 \
#   --add_context_role_text


### 10. onlyone_step_prompt——scannet
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/onlyone_step_prompt_scannet_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/onlyone_step_prompt/scannet/step_mix_onlyone0227_0015000 \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/models/ft_weights_old/step_mix_onlyone0227_0015000 \
#   --add_context_role_text


### 11. onlyone_step_prompt——dl3dv
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/onlyone_step_prompt_dl3dv_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/onlyone_step_prompt/dl3dv/step_mix_onlyone0227_0015000 \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/models/ft_weights_old/step_mix_onlyone0227_0015000 \
#   --add_context_role_text


### 12. onlyone_step_prompt——matterport3d
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/onlyone_step_prompt_matterport3d_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/onlyone_step_prompt/matterport3d/step_mix_onlyone0227_0015000 \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/models/ft_weights_old/step_mix_onlyone0227_0015000 \
#   --add_context_role_text


### 13. refine_step_prompt——scannet
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/refine_step_prompt_scannet_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/refine_step_prompt/scannet/bagel \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT \
#   --add_context_role_text


### 14. refine_step_prompt——dl3dv
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/refine_step_prompt_dl3dv_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/refine_step_prompt/dl3dv/bagel \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT \
#   --add_context_role_text


### 15. refine_step_prompt——matterport3d
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/refine_step_prompt_matterport3d_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/refine_step_prompt/matterport3d/bagel \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT \
#   --add_context_role_text



### 16. step_prompt——scannet
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/step_prompt_scannet_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/step_prompt/scannet/bagel \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT \
#   --add_context_role_text


### 17. step_prompt——dl3dv
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/step_prompt_dl3dv_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/step_prompt/dl3dv/bagel \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT \
#   --add_context_role_text


### 18. step_prompt——matterport3d
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/step_prompt_matterport3d_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/step_prompt/matterport3d/bagel \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT \
#   --add_context_role_text


### 19. onlyone_refine_step_prompt——scannet
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/onlyone_refine_step_prompt_scannet_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/onlyone_refine_step_prompt/scannet/bagel \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT \
#   --add_context_role_text


### 20. onlyone_refine_step_prompt——dl3dv
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/onlyone_refine_step_prompt_dl3dv_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/onlyone_refine_step_prompt/dl3dv/bagel \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT \
#   --add_context_role_text


### 21. onlyone_refine_step_prompt——matterport3d
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/onlyone_refine_step_prompt_matterport3d_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/onlyone_refine_step_prompt/matterport3d/bagel \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT \
#   --add_context_role_text


### 22. onlyone_step_prompt——scannet
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/onlyone_step_prompt_scannet_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/onlyone_step_prompt/scannet/bagel \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT \
#   --add_context_role_text


### 23. onlyone_step_prompt——dl3dv
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/onlyone_step_prompt_dl3dv_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/onlyone_step_prompt/dl3dv/bagel \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT \
#   --add_context_role_text


### 24. onlyone_step_prompt——matterport3d
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/onlyone_step_prompt_matterport3d_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/onlyone_step_prompt/matterport3d/bagel \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT \
#   --add_context_role_text


### 1. force_onlyone_step_prompt——scannet
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/force_onlyone_step_prompt_scannet_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/force_onlyone_step_prompt/scannet/bagel \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT \



### 2. force_onlyone_step_prompt——dl3dv
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/force_onlyone_step_prompt_dl3dv_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/force_onlyone_step_prompt/dl3dv/bagel \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT \



### 3. force_onlyone_step_prompt——matterport3d
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/force_onlyone_step_prompt_matterport3d_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/force_onlyone_step_prompt/matterport3d/bagel \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT \


### 4. force_step_prompt——scannet
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/force_step_prompt_scannet_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/force_step_prompt/scannet/bagel \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT \



### 5. force_step_prompt——dl3dv
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/force_step_prompt_dl3dv_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/force_step_prompt/dl3dv/bagel \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT \



### 6. force_step_prompt——matterport3d
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/force_step_prompt_matterport3d_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/force_step_prompt/matterport3d/bagel \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT \




### 7. force_onlyone_step_prompt——scannet
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/force_onlyone_step_prompt_scannet_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/force_onlyone_step_prompt/scannet/step_mix_onlyone_force0303_0020000 \
#   --model_path /mnt/inspurfs/efm_t/linjingli/model/bagel_merge/step_mix_onlyone_force0303_0020000 \



### 8. force_onlyone_step_prompt——dl3dv
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/force_onlyone_step_prompt_dl3dv_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/force_onlyone_step_prompt/dl3dv/step_mix_onlyone_force0303_0020000 \
#   --model_path /mnt/inspurfs/efm_t/linjingli/model/bagel_merge/step_mix_onlyone_force0303_0020000 \



### 9. force_onlyone_step_prompt——matterport3d
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/force_onlyone_step_prompt_matterport3d_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/force_onlyone_step_prompt/matterport3d/step_mix_onlyone_force0303_0020000 \
#   --model_path /mnt/inspurfs/efm_t/linjingli/model/bagel_merge/step_mix_onlyone_force0303_0020000  \


### 10. force_step_prompt——scannet
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/force_step_prompt_scannet_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/force_step_prompt/scannet/step_mix_force0303_0020000 \
#   --model_path /mnt/inspurfs/efm_t/linjingli/model/bagel_merge/step_mix_force0303_0020000 \



### 11. force_step_prompt——dl3dv
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/force_step_prompt_dl3dv_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/force_step_prompt/dl3dv/step_mix_force0303_0020000 \
#   --model_path /mnt/inspurfs/efm_t/linjingli/model/bagel_merge/step_mix_force0303_0020000 \



### 12. force_step_prompt——matterport3d
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/force_step_prompt_matterport3d_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/force_step_prompt/matterport3d/step_mix_force0303_0020000 \
#   --model_path /mnt/inspurfs/efm_t/linjingli/model/bagel_merge/step_mix_force0303_0020000 \


### 13. force_onlyone_step_prompt——scannet
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/force_onlyone_step_prompt_scannet_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/force_onlyone_step_prompt/scannet/step_mix_force0303_0020000 \
#   --model_path   /mnt/inspurfs/efm_t/linjingli/model/bagel_merge/step_mix_force0303_0020000 \



### 14. force_onlyone_step_prompt——dl3dv
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/force_onlyone_step_prompt_dl3dv_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/force_onlyone_step_prompt/dl3dv/step_mix_force0303_0020000 \
#   --model_path /mnt/inspurfs/efm_t/linjingli/model/bagel_merge/step_mix_force0303_0020000 \



### 15. force_onlyone_step_prompt——matterport3d
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/force_onlyone_step_prompt_matterport3d_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/force_onlyone_step_prompt/matterport3d/step_mix_force0303_0020000 \
#   --model_path /mnt/inspurfs/efm_t/linjingli/model/bagel_merge/step_mix_force0303_0020000  \





### 1. rule_base_simple_2context——scannet
python batch_infer.py \
  --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/rule_base_simple_2context_scannet_test.jsonl \
  --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
  --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/rule_base_simple_2context/scannet/bagel \
  --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT \



### 2. rule_base_simple_2context——dl3dv
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/rule_base_simple_2context_dl3dv_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/rule_base_simple_2context/dl3dv/bagel \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT \



### 3. rule_base_simple_2context——matterport3d
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/rule_base_simple_2context_matterport3d_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/rule_base_simple_2context/matterport3d/bagel \
#   --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT \


# ## 4. rule_base_simple_2context——scannet——dropout
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/rule_base_simple_2context_scannet_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/rule_base_simple_2context/scannet/rule_base_simple_2context_3data_0306_0015000 \
#   --model_path /mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/ft_weights/rule_base_simple_2context_3data_0306_0015000 \



# ## 5. rule_base_simple_2context——dl3dv——dropout
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/rule_base_simple_2context_dl3dv_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/rule_base_simple_2context/dl3dv/rule_base_simple_2context_3data_0306_0015000 \
#   --model_path /mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/ft_weights/rule_base_simple_2context_3data_0306_0015000 \



# ## 6. rule_base_simple_2context——matterport3d——dropout
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/rule_base_simple_2context_matterport3d_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/rule_base_simple_2context/matterport3d/rule_base_simple_2context_3data_0306_0015000 \
#   --model_path /mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/ft_weights/rule_base_simple_2context_3data_0306_0015000 \


# ## 7. rule_base_simple_2context——scannet——nodropout
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/rule_base_simple_2context_scannet_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/rule_base_simple_2context/scannet/rule_base_simple_2context_3data_0308_0015000 \
#   --model_path /mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/ft_weights/rule_base_simple_2context_3data_0308_0015000 \



# ## 8. rule_base_simple_2context——dl3dv——nodropout
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/rule_base_simple_2context_dl3dv_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/rule_base_simple_2context/dl3dv/rule_base_simple_2context_3data_0308_0015000 \
#   --model_path /mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/ft_weights/rule_base_simple_2context_3data_0308_0015000 \



# ## 9. rule_base_simple_2context——matterport3d——nodropout
# python batch_infer.py \
#   --eval-json /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/rule_base_simple_2context_matterport3d_test.jsonl \
#   --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
#   --out-dir /mnt/inspurfs/efm_t/longyilin/genspace/outputs/rule_base_simple_2context/matterport3d/rule_base_simple_2context_3data_0308_0015000 \
#   --model_path /mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/ft_weights/rule_base_simple_2context_3data_0308_0015000  \
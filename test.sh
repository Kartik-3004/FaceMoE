# CUDA_VISIBLE_DEVICES=0 python /cis/home/knaraya4/FaceMoE/validation_hq/validate_hq.py \
#     --model_load_path /cis/home/knaraya4/FaceMoE/weights/swin4m_tinyface_exp_3_k_2_full/model.pt \
#     --model_type swin_moe \
#     --batch_size 1400 \
#     --num_experts 3 \
#     --k 2 \
#     --image_size 120

# CUDA_VISIBLE_DEVICES=1 python /cis/home/knaraya4/FaceMoE/validation_lq/validate_tinyface.py \
#     --model_load_path /cis/home/knaraya4/FaceMoE/weights/swin4m_tinyface_exp_3_k_2_full/model.pt \
#     --model_type swin_moe \
#     --batch_size 1200 \
#     --num_experts 3 \
#     --k 2 \
#     --image_size 120

# CUDA_VISIBLE_DEVICES=0 python /cis/home/knaraya4/FaceMoE/validation_ijb/eval_ijb.py \
#     --model_load_path /cis/home/knaraya4/FaceMoE/weights/swin4m_tinyface_exp_3_k_2_full/model.pt \
#     --data_root /cis/home/knaraya4/data/ijb/IJBB/ \
#     --model_type swin_moe \
#     --batch-size 1024 \
#     --num_experts 3 \
#     --k 2 \
#     --target IJBB

# CUDA_VISIBLE_DEVICES=1 python /cis/home/knaraya4/FaceMoE/validation_ijb/eval_ijb.py \
#     --model_load_path /cis/home/knaraya4/FaceMoE/weights/swin4m_tinyface_exp_3_k_2_full/model.pt \
#     --data_root /cis/home/knaraya4/data/ijb/IJBC/ \
#     --model_type swin_moe \
#     --batch-size 1024 \
#     --num_experts 3 \
#     --k 2 \
#     --target IJBC
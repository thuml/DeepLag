export CUDA_VISIBLE_DEVICES=5


# bounded NS
python test_bc_h.py \
  --ckpt-dir ./checkpoints \
  --dataset-nickname bc \
  --data-path /home/miaoshangchen/NAS/Bounded_NS \
  --model-name UNet_2D \
  --time-str 20240519_172142 \
  --milestone best \
  --T-out 30

python test_bc_h.py \
  --ckpt-dir ./checkpoints \
  --dataset-nickname bc \
  --data-path /home/miaoshangchen/NAS/Bounded_NS \
  --model-name FNO_2D \
  --time-str 20240520_093530 \
  --milestone best \
  --T-out 30

python test_bc_h.py \
  --ckpt-dir ./checkpoints \
  --dataset-nickname bc \
  --data-path /home/miaoshangchen/NAS/Bounded_NS \
  --model-name GkTrm_2D \
  --time-str 20240520_013510 \
  --milestone best \
  --T-out 30

python test_bc_h.py \
  --ckpt-dir ./checkpoints \
  --dataset-nickname bc \
  --data-path /home/miaoshangchen/NAS/Bounded_NS \
  --model-name GNOT_2D \
  --time-str 20240429_065015 \
  --milestone best \
  --T-out 30

python test_bc_h.py \
  --ckpt-dir ./checkpoints \
  --dataset-nickname bc \
  --data-path /home/miaoshangchen/NAS/Bounded_NS \
  --model-name LSM_2D \
  --time-str 20240520_094033 \
  --milestone best \
  --T-out 30

python test_bc_h.py \
  --ckpt-dir ./checkpoints \
  --dataset-nickname bc \
  --data-path /home/miaoshangchen/NAS/Bounded_NS \
  --model-name Factformer_2D \
  --time-str 20240429_064817 \
  --milestone best \
  --T-out 30

python test_bc_h_vortex.py \
  --ckpt-dir ./checkpoints \
  --dataset-nickname bc \
  --data-path /home/miaoshangchen/NAS/Bounded_NS \
  --model-name Vortex_2D \
  --time-str 20240520_055530 \
  --milestone best \
  --T-out 30

python -u test_bc_h.py \
  --ckpt-dir ./checkpoints \
  --dataset-nickname bc \
  --data-path /home/miaoshangchen/NAS/Bounded_NS \
  --model-name DeepLag_2D \
  --time-str 20240520_140626 \
  --milestone best \
  --T-out 30


# Ocean Current
python test_sea_h.py \
  --ckpt-dir ./checkpoints \
  --dataset-nickname sea \
  --model-name UNet_2D \
  --time-str 20240520_132532 \
  --milestone best \
  --T-out 30

python test_sea_h.py \
  --ckpt-dir ./checkpoints \
  --dataset-nickname sea \
  --model-name FNO_2D \
  --time-str 20240520_020333 \
  --milestone best \
  --T-out 30

python test_sea_h.py \
  --ckpt-dir ./checkpoints \
  --dataset-nickname sea \
  --model-name GkTrm_2D \
  --time-str 20240520_020707 \
  --milestone best \
  --T-out 30

python test_sea_h.py \
  --ckpt-dir ./checkpoints \
  --dataset-nickname sea \
  --model-name GNOT_2D \
  --time-str 20240501_154700 \
  --milestone best \
  --T-out 30

python test_sea_h.py \
  --ckpt-dir ./checkpoints \
  --dataset-nickname sea \
  --model-name LSM_2D \
  --time-str 20240520_132354 \
  --milestone best \
  --T-out 30

python test_sea_h.py \
  --ckpt-dir ./checkpoints \
  --dataset-nickname sea \
  --model-name Factformer_2D \
  --time-str 20240501_152629 \
  --milestone best \
  --T-out 30

python test_sea_h_vortex.py \
  --ckpt-dir ./checkpoints \
  --dataset-nickname sea \
  --model-name Vortex_2D \
  --time-str 20240522_023056 \
  --milestone best \
  --T-out 30

python test_sea_h.py \
  --ckpt-dir ./checkpoints \
  --dataset-nickname sea \
  --model-name DeepLag_2D \
  --time-str 20240507_170237 \
  --milestone best \
  --T-out 30


# Smoke
python test_smoke_h.py \
  --ckpt-dir ./checkpoints \
  --dataset-nickname smoke \
  --model-name UNet_3D \
  --time-str 20240520_064624 \
  --milestone best \
  --T-out 30

python test_smoke_h.py \
  --ckpt-dir ./checkpoints \
  --dataset-nickname smoke \
  --model-name FNO_3D \
  --time-str 20240520_064910 \
  --milestone best \
  --T-out 30

python test_smoke_h.py \
  --ckpt-dir ./checkpoints \
  --dataset-nickname smoke \
  --model-name GkTrm_3D \
  --time-str 20240501_160526 \
  --milestone best \
  --T-out 30

python test_smoke_h.py \
  --ckpt-dir ./checkpoints \
  --dataset-nickname smoke \
  --model-name GNOT_3D \
  --time-str 20240503_060448 \
  --milestone best \
  --T-out 30

python test_smoke_h.py \
  --ckpt-dir ./checkpoints \
  --dataset-nickname smoke \
  --model-name LSM_3D \
  --time-str 20240520_063957 \
  --milestone best \
  --T-out 30

python test_smoke_h.py \
  --ckpt-dir ./checkpoints \
  --dataset-nickname smoke \
  --model-name Factformer_3D \
  --time-str 20240520_173349 \
  --milestone best \
  --T-out 30

python test_smoke_h.py \
  --ckpt-dir ./checkpoints \
  --dataset-nickname smoke \
  --model-name DeepLag_3D \
  --time-str 20240520_172807 \
  --milestone best \
  --T-out 30

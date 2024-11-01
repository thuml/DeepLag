export CUDA_VISIBLE_DEVICES=4

python exp_sea_h.py \
  --dataset-nickname sea \
  --data-path /home/miaoshangchen/NAS/sea_data_small/data_sea \
  --region kuroshio \
  --ntrain 3000 \
  --ntest  600 \
  --ntotal 3600 \
  --in-dim 10 \
  --out-dim 1 \
  --in-var 5 \
  --out-var 5 \
  --has-t \
  --tmin 0 \
  --tmax 9 \
  --h 180 \
  --w 300 \
  --h-down 1 \
  --w-down 1 \
  --T-in 10 \
  --T-out 10 \
  --fill-value \-32760 \
  --batch-size 10 \
  --learning-rate 0.0005 \
  --epochs 101 \
  --step-size 100 \
  --model FNO_2D \
  --model-nickname fno \
  --d-model 64 \
  --num-samples 512 \
  --num-layers 12 \
  --num-basis 12 \
  --num-token 4 \
  --patch-size 4,4 \
  --padding 0,0 \
  --kernel-size 3 \
  --offset-ratio-range 16,8 \
  --resample-strategy learned \
  --model-save-path ./checkpoints/sea \
  --model-save-name fno.pt
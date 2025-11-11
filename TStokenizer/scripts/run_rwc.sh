DATASET=RWC

python main.py \
--save_path ../vqvae/$DATASET \
--dataset whale \
--data_path ../datasets/$DATASET \
--n_embed 384 \
--wave_length 32 \
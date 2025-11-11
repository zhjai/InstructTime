DATASET=FD

python main.py \
--save_path ../vqvae/$DATASET \
--dataset dev \
--data_path ../datasets/$DATASET \
--n_embed 512 \
--wave_length 40 \
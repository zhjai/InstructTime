DATASET=EEG

python main.py \
--save_path ../vqvae/$DATASET \
--dataset sleep \
--data_path ../datasets/$DATASET \
--n_embed 256 \
--wave_length 25 \
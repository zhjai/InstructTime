DATASET=GEO

python main.py \
--save_path ../vqvae/$DATASET \
--dataset geo \
--data_path ../datasets/$DATASET \
--n_embed 128 \
--wave_length 40 \
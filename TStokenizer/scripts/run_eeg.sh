DATASET=EEG

python main.py \
--save_path ../vqvae/$DATASET \
--dataset eeg \
--data_path ../datasets/$DATASET \
--n_embed 256 \
--wave_length 25 \
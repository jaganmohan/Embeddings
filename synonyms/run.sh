#!/bin/sh

# Step 1: Preprocess data
## Modify the below line and add parameters accordingly
#python preprocess_raw.py \
    
    
# Step 2: Train model over data and save the embeddings    
python ml/model.py \
    --train-bs=128 \
    --emb-size=256 \
    --skip-window=1 \
    --num-skips=2 \
    --neg-samples=64 \
    --epochs=100001 \
    --file=data/text8.zip \
    --tsne-img-file=output/embeddings.png \
    --emb-file=output/embeddings.txt \
    --log-dir=log \

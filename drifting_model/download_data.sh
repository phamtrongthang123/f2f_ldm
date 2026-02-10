#!/bin/bash
mkdir -p data
cd data

BASE_URL="https://diffusion-policy.cs.columbia.edu/data/training"

download_and_unzip() {
    FILE=$1
    if [ ! -d "${FILE%.zip}" ] && [ ! -f "${FILE%.zip}.zarr" ] && [ ! -d "${FILE%.zip}.zarr" ]; then
        echo "Downloading $FILE..."
        wget "$BASE_URL/$FILE"
        unzip "$FILE"
        rm "$FILE"
    else
        echo "$FILE already exists, skipping."
    fi
}

download_and_unzip "pusht.zip"
download_and_unzip "block_pushing.zip"
download_and_unzip "kitchen.zip"
download_and_unzip "pusht_real.zip"
download_and_unzip "robomimic_lowdim.zip"
# download_and_unzip "robomimic_image.zip" # Warning: 78 GB 

cd ..
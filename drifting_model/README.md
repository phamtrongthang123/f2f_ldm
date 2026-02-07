conda create -n drifting -y 
conda activate drifting 
pip install uv


Setup Steps
1. Download: Register and download ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar.
2. Extract: Use a script (like this common one
    (https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh)) to extract the images into the
    subfolder structure shown above.
3. Configure: Update the IMAGENET_DIR variable in scripts/slurm_train.sh to point to your extracted data, or pass it
    via the --imagenet_dir flag when running train_imagenet.py.


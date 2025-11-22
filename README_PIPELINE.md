# m_3 → m_1 Ultrasound Translation Pipeline

**Goal**: Translate m_3 ultrasound mode to m_1 mode using PyTorch Lightning

## Pipeline Overview

```
m_3 images  →  SDXL + Embedding Translation  →  m_1 style images
```

---

## Quick Start (5 Steps)

### Step 1: Prepare Dataset
```bash
bash 1_prepare_data.sh
```
- Crops ultrasound ROI from raw frames
- Resizes to 512×512
- Creates `data/ultrasound_dataset/train/m1/` and `m3/`

---

### Step 2: Train SDXL (PyTorch Lightning)
```bash
sbatch 2_train_sdxl.sh
```
- Trains on **combined** m_1 + m_3 data (learns both domains)
- Uses PyTorch Lightning (no Accelerate)
- Takes ~8-12 hours on 1 GPU
- Checkpoints: `checkpoints/m3_to_m1/`

**Monitor:**
```bash
squeue -u $USER
tail -f slurm_logs/<job_id>.out
```

---

### Step 3: Extract DINO Features
```bash
bash 3_extract_features.sh
```
- Extracts 384-dim embeddings using DINOv2
- Creates `data/ultrasound_dataset/features/trainA/` (m_1) and `trainB/` (m_3)
- Takes ~5 minutes

---

### Step 4: Train Embedding Translation (m_3 → m_1)
```bash
sbatch 4_train_embedding_translation.sh
```
- **Automatically reverses** trainA/trainB (trainA=m_3, trainB=m_1)
- Trains CycleGAN in embedding space
- Takes ~30-60 minutes
- Checkpoint: `embedding_translation/checkpoints/m3_to_m1.ckpt`

---

### Step 5: Run Inference
```bash
bash 5_run_inference.sh
```
- Translates m_3 test images → m_1 style
- Results: `results/m3_to_m1/`
- Takes ~1 second per image

---

## File Structure

```
f2f_ldm/
├── 1_prepare_data.sh              # Step 1: Data prep
├── 2_train_sdxl.sh                # Step 2: SDXL training (SLURM)
├── 3_extract_features.sh          # Step 3: Extract DINO features
├── 4_train_embedding_translation.sh # Step 4: ET training (SLURM)
├── 5_run_inference.sh             # Step 5: Inference
│
├── train_pl.py                    # PyTorch Lightning training code
├── setup_ultrasound_dataset.py    # Data preprocessing
├── extract_features_ultrasound.py # Feature extraction
├── feature_extractor_ultrasound.py # DINOv2 wrapper
│
├── data/ultrasound_dataset/
│   ├── train/m1/                  # Processed m_1 frames
│   ├── train/m3/                  # Processed m_3 frames
│   ├── features/                  # DINO features
│   └── test/m3/                   # Test images
│
├── checkpoints/m3_to_m1/          # SDXL checkpoints
├── embedding_translation/checkpoints/m3_to_m1.ckpt
└── results/m3_to_m1/              # Translated images
```

---

## Key Points

### Why SDXL trains on both domains?
- SDXL learns to denoise **both** m_1 and m_3 images
- Embedding translation (CycleGAN) controls the direction

### Why swap trainA/trainB for m_3 → m_1?
- Original: trainA=m_1, trainB=m_3 → learns m_1→m_3
- Reversed: trainA=m_3, trainB=m_1 → learns m_3→m_1
- Step 4 script handles this automatically!

### PyTorch Lightning vs Accelerate
- No `accelerate launch` needed
- Cleaner code structure
- Built-in checkpointing, logging, multi-GPU
- Easier debugging

---

## Tuning Translation Quality

Edit `5_run_inference.sh` to adjust:

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `HIGH_NOISE_FRAC` | 0.5 | 0.3-0.7 | Higher = more aggressive |
| `GUIDANCE_SCALE` | 12.0 | 8.0-15.0 | Higher = stronger m_1 style |
| `ET_WEIGHT` | 0.25 | 0.0-1.0 | Higher = more ET influence |
| `REG` | l0 | l0/l1 | l0=sparse, l1=smooth |

---

## Timeline

| Step | Time | GPU |
|------|------|-----|
| 1. Data prep | 5 min | No |
| 2. SDXL training | 8-12 hours | Yes |
| 3. Feature extraction | 5 min | Optional |
| 4. ET training | 30-60 min | Yes |
| 5. Inference | 1 sec/image | Yes |

**Total first run**: ~10-14 hours

---

## Troubleshooting

### SLURM job fails
- Check logs: `cat slurm_logs/<job_id>.err`
- Verify conda env: `conda activate f2fldm`
- Check GPU availability: `squeue -p gpu`

### OOM errors
- Reduce batch size in `2_train_sdxl.sh` (change `--batch_size=2` to `1`)
- Increase `--accumulate_grad_batches`

### Poor translation quality
- Train longer (increase `--max_steps`)
- Tune inference parameters (see table above)
- Check if data cropping is correct

---

## Clean Pipeline - No Clutter!

Only 5 numbered scripts + core Python files. That's it! 🎯

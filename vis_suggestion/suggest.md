2D ZEA synthesis (reproduce helpers/zea_synthesize_dataset.py)

- Role: sample scatterers → simulate_rf → Beamform → compand → save NPZ.
- Visual checks that match those steps:
    - Scatterer distribution: plot x vs z from _random_scatterers to verify z_bias for tissue vs haze.
    - Beamformed RF frame (pre‑compand): imshow(frame, cmap="gray") and a log‑compressed view 20*log10(abs(frame)+eps) to see
    B‑mode‑like contrast.
    - Companding effect: histogram before/after _normalize_and_compand to ensure [0,1] then scaled [0,255].
    - Tissue vs haze: side‑by‑side frames or frame_tissue - frame_haze to check structured haze dominance near field.

3D ZEA synthesis (reprodcue 3D data/helpers/zea_synthesize_dataset_3d.py)

- Role: one 3D scatterer cloud → slice weighting (_slice_weighted_magnitudes) → per‑slice Beamform → stack to volume.
- Visual checks:
    - Slice weighting curve: plot Gaussian weights vs y to validate slice_sigma.
    - Slice montage: grid of slices from one volume to ensure smooth variation across y.
    - Slice energy vs index: mean(|slice|) across slices to catch dead slices or wrong ylims.
    - Volume browsing: use napari to scroll volume[s] if you want quick interactive sanity.

Config updater (reproduce helpers/update_zea_inference_config.py)

- Role: choose most recent wandb runs (zea_tissue/zea_haze) and write them into the inference YAML.
- Visual verification (text‑centric, but essential here):
    - Diff the config before/after (git diff or yq) to confirm run_id.sgm and sgm.corruptor_run_id.
    - List latest wandb configs (ls -lt dehazing-diffusion/joint_diffusion/wandb/**/files/config.yaml) to confirm it picked
    the right run.

Dataset loader + corruptor (dehazing-diffusion/joint_diffusion/datasets.py, utils/corruptors.py)

- Role: load zea_synth NPZ, then corrupt with haze (y = x + alpha * n).
- Best built‑in visualization:
    - inference.py -t show_dataset calls EvalDataset.plot_batch and shows clean / corrupted / noise in a grid when
    paired_data: true.
    - This directly validates HazeCorruptor behavior (noise image and haze_strength scaling).

Inference + guidance (dehazing-diffusion/joint_diffusion/inference.py, generators/SGM/guidance.py, utils/inverse.py)

- Role: joint sampling with PIGDM; optionally track intermediate states.
- Visual checks:
    - Set keep_track: true in the inference config; the denoiser saves an animation to dehazing-diffusion/joint_diffusion/
    figures/*_animation.gif.
    - Use plot_results / run_metrics tasks to generate side‑by‑side comparisons and metrics in the eval folder
    (comparison.pdf/png, metrics.pdf/png).

Ultrasound‑specific stats (dehazing-diffusion/processing.py)

- Role: speckle/statistical diagnostics.
- Visual checks:
    - ks_test(sample, data, plot=True) to compare speckle distributions (dehazed vs GT).
    - get_fhwm_from_autocorrelation to check lateral/axial resolution changes.
    - histogram_match / equalize_histogram to standardize brightness for fair visual comparison.
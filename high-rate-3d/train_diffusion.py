"""
train_diffusion.py — Train the ZEA DiffusionModel on echo data.

Custom JAX training loop for the diffusion model using optax for
optimization. Trains on EchoNet-Dynamic (or CAMUS) frames, with EMA
weight tracking, periodic checkpointing, and sample generation.

The trained model is saved in ZEA preset format and can be loaded with:
    model = DiffusionModel.from_preset("outputs/training/final_model")
"""

import env_setup  # noqa: F401 — must be first

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
from zea import init_device
from zea.models.diffusion import DiffusionModel
from zea.visualize import plot_image_grid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train diffusion model on echo data")
    p.add_argument("--data-dir", default="EchoNet-Dynamic",
                    help="Dir with train_frames.npy / val_frames.npy (from convert_videos.py)")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--ema-val", type=float, default=0.999)
    p.add_argument("--checkpoint-every", type=int, default=5,
                    help="Save checkpoint every N epochs")
    p.add_argument("--sample-every", type=int, default=5,
                    help="Generate samples every N epochs")
    p.add_argument("--n-sample-steps", type=int, default=50,
                    help="Diffusion steps for sample generation")
    p.add_argument("--resume", default=None,
                    help="Checkpoint directory to resume from")
    p.add_argument("--output-dir", default="outputs/training")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(npy_path):
    """Memory-map a .npy file of uint8 frames. Instant, zero RAM."""
    data = np.load(npy_path, mmap_mode='r')  # (N, 112, 112) uint8
    print(f"  {npy_path}: {data.shape}, dtype={data.dtype}")
    return data


def get_batch(data, indices):
    """Read a batch from mmap, convert to float32 [-1, 1] with channel dim."""
    batch = data[indices].astype(np.float32) / 127.5 - 1.0
    return batch[..., np.newaxis]  # (B, H, W, 1)


# ---------------------------------------------------------------------------
# Pure-function diffusion schedule (JIT-friendly)
# ---------------------------------------------------------------------------

def diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    """Cosine diffusion schedule — pure JAX, no model state."""
    start_angle = jnp.arccos(max_signal_rate)
    end_angle = jnp.arccos(min_signal_rate)
    angles = start_angle + diffusion_times * (end_angle - start_angle)
    signal_rates = jnp.cos(angles)
    noise_rates = jnp.sin(angles)
    return noise_rates, signal_rates


# ---------------------------------------------------------------------------
# EMA update
# ---------------------------------------------------------------------------

@jax.jit
def ema_update(ema_params, train_params, ema_val=0.999):
    return jax.tree.map(
        lambda e, t: ema_val * e + (1 - ema_val) * t,
        ema_params, train_params,
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_for_inference(model, save_dir):
    """Swap EMA weights into the main network, save preset, swap back."""
    originals = [w.value.copy() for w in model.network.weights]
    for w, ema_w in zip(model.network.weights, model.ema_network.weights):
        w.assign(ema_w.value)
    model.save_to_preset(save_dir)
    for w, orig in zip(model.network.weights, originals):
        w.assign(orig)


def save_training_state(path, trainable_vars, non_trainable_vars, ema_vars,
                        opt_state, epoch):
    """Save raw training state as numpy arrays for resumption."""
    os.makedirs(path, exist_ok=True)
    trainable_np = jax.tree.map(np.array, trainable_vars)
    non_trainable_np = jax.tree.map(np.array, non_trainable_vars)
    ema_np = jax.tree.map(np.array, ema_vars)
    opt_np = jax.tree.map(np.array, opt_state)
    np.savez(os.path.join(path, "train_state.npz"),
             **{f"t_{i}": v for i, v in enumerate(jax.tree.leaves(trainable_np))})
    np.savez(os.path.join(path, "non_trainable_state.npz"),
             **{f"nt_{i}": v for i, v in enumerate(jax.tree.leaves(non_trainable_np))})
    np.savez(os.path.join(path, "ema_state.npz"),
             **{f"e_{i}": v for i, v in enumerate(jax.tree.leaves(ema_np))})
    # Optimizer state has nested structure — pickle it
    import pickle
    with open(os.path.join(path, "opt_state.pkl"), "wb") as f:
        pickle.dump(opt_np, f)
    with open(os.path.join(path, "epoch.txt"), "w") as f:
        f.write(str(epoch))


def load_training_state(path, trainable_vars, non_trainable_vars, ema_vars,
                        opt_state):
    """Load training state from a checkpoint directory."""
    import pickle

    t_data = np.load(os.path.join(path, "train_state.npz"))
    t_leaves = [t_data[f"t_{i}"] for i in range(len(t_data.files))]
    trainable_vars = jax.tree.unflatten(jax.tree.structure(trainable_vars), t_leaves)

    nt_path = os.path.join(path, "non_trainable_state.npz")
    if os.path.exists(nt_path):
        nt_data = np.load(nt_path)
        nt_leaves = [nt_data[f"nt_{i}"] for i in range(len(nt_data.files))]
        non_trainable_vars = jax.tree.unflatten(
            jax.tree.structure(non_trainable_vars), nt_leaves)

    e_data = np.load(os.path.join(path, "ema_state.npz"))
    e_leaves = [e_data[f"e_{i}"] for i in range(len(e_data.files))]
    ema_vars = jax.tree.unflatten(jax.tree.structure(ema_vars), e_leaves)

    with open(os.path.join(path, "opt_state.pkl"), "rb") as f:
        opt_state = pickle.load(f)
    # Convert numpy arrays back to JAX arrays
    opt_state = jax.tree.map(jnp.array, opt_state)

    with open(os.path.join(path, "epoch.txt")) as f:
        start_epoch = int(f.read().strip())

    return trainable_vars, non_trainable_vars, ema_vars, opt_state, start_epoch


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

def generate_and_save_samples(model, epoch, output_dir, n_steps):
    """Generate 8 unconditional samples using EMA network and save as grid."""
    samples = model.sample(n_samples=8, n_steps=n_steps, verbose=False)
    samples_np = np.array(samples)
    images = [samples_np[i, :, :, 0] for i in range(samples_np.shape[0])]
    fig, _ = plot_image_grid(images, ncols=4, cmap="gray", vmin=-1, vmax=1)
    fig.suptitle(f"Samples — Epoch {epoch}")
    path = os.path.join(output_dir, f"samples_epoch_{epoch:03d}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved samples to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    init_device(verbose=False)
    print(f"JAX devices: {jax.devices()}")

    # ---- Model ----
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = DiffusionModel.from_preset(
            os.path.join(args.resume, "model_preset"))
    else:
        model = DiffusionModel(
            input_shape=(112, 112, 1),
            input_range=(-1, 1),
            network_name="unet_time_conditional",
            network_kwargs={"block_depth": 2, "widths": [32, 64, 96, 128]},
            min_signal_rate=0.02,
            max_signal_rate=0.95,
            ema_val=args.ema_val,
            guidance=None,
            operator=None,
        )
    img_shape = model.input_shape[:2]  # (112, 112)
    print(f"Model input shape: {model.input_shape}")

    # ---- Data (memory-mapped, instant) ----
    print(f"Loading data from {args.data_dir} ...")
    train_data = load_data(os.path.join(args.data_dir, "train_frames.npy"))
    val_data = load_data(os.path.join(args.data_dir, "val_frames.npy"))

    # ---- Optimizer ----
    optimizer = optax.adamw(learning_rate=args.lr, weight_decay=args.weight_decay)

    # Extract variable lists for stateless_call
    trainable_vars = [w.value for w in model.network.trainable_weights]
    non_trainable_vars = [w.value for w in model.network.non_trainable_weights]
    ema_vars = [w.value for w in model.ema_network.weights]
    opt_state = optimizer.init(trainable_vars)

    # Build index map: model.network.weights order → trainable/non-trainable
    # Needed to merge vars for EMA update (which tracks ALL weights)
    _trainable_ids = {id(w) for w in model.network.trainable_weights}
    _is_trainable = [id(w) in _trainable_ids for w in model.network.weights]

    def _merge_vars(t_vars, nt_vars):
        """Merge trainable + non-trainable vars in model.network.weights order."""
        result = []
        ti, ni = 0, 0
        for is_t in _is_trainable:
            if is_t:
                result.append(t_vars[ti])
                ti += 1
            else:
                result.append(nt_vars[ni])
                ni += 1
        return result

    start_epoch = 0
    if args.resume:
        (trainable_vars, non_trainable_vars, ema_vars,
         opt_state, start_epoch) = load_training_state(
            args.resume, trainable_vars, non_trainable_vars, ema_vars, opt_state)
        print(f"  Resumed from epoch {start_epoch}")

    # ---- Build JIT-compiled train step ----
    min_t = float(model.min_t)
    max_t = float(model.max_t)
    min_signal_rate = float(model.min_signal_rate)
    max_signal_rate = float(model.max_signal_rate)

    @jax.jit
    def train_step(trainable_vars, non_trainable_vars, opt_state, batch, rng_key):
        def loss_fn(tv):
            rng_noise, rng_time = jax.random.split(rng_key)
            batch_size = batch.shape[0]

            noises = jax.random.normal(rng_noise, shape=batch.shape)
            diffusion_times = jax.random.uniform(
                rng_time, shape=(batch_size, 1, 1, 1),
                minval=min_t, maxval=max_t,
            )

            noise_rates, signal_rates = diffusion_schedule(
                diffusion_times, min_signal_rate, max_signal_rate)
            noisy_data = signal_rates * batch + noise_rates * noises

            pred_noises, updated_non_trainable = model.network.stateless_call(
                tv, non_trainable_vars,
                [noisy_data, noise_rates ** 2], training=True,
            )
            pred_images = (noisy_data - noise_rates * pred_noises) / signal_rates

            noise_loss = jnp.mean((noises - pred_noises) ** 2)
            image_loss = jnp.mean((batch - pred_images) ** 2)
            return noise_loss, (image_loss, updated_non_trainable)

        (noise_loss, (image_loss, new_non_trainable)), grads = jax.value_and_grad(
            loss_fn, has_aux=True,
        )(trainable_vars)

        updates, new_opt_state = optimizer.update(grads, opt_state, trainable_vars)
        new_trainable = optax.apply_updates(trainable_vars, updates)

        return new_trainable, new_non_trainable, new_opt_state, noise_loss, image_loss

    # Validation uses EMA network (matching DiffusionModel.test_step which
    # calls self.denoise() → self() with training=False → ema_network).
    # ema_network.trainable=False so all weights are non-trainable.
    @jax.jit
    def val_step(ema_vars, batch, rng_key):
        rng_noise, rng_time = jax.random.split(rng_key)
        batch_size = batch.shape[0]

        noises = jax.random.normal(rng_noise, shape=batch.shape)
        diffusion_times = jax.random.uniform(
            rng_time, shape=(batch_size, 1, 1, 1),
            minval=min_t, maxval=max_t,
        )

        noise_rates, signal_rates = diffusion_schedule(
            diffusion_times, min_signal_rate, max_signal_rate)
        noisy_data = signal_rates * batch + noise_rates * noises

        pred_noises, _ = model.ema_network.stateless_call(
            [], ema_vars,
            [noisy_data, noise_rates ** 2], training=False,
        )
        pred_images = (noisy_data - noise_rates * pred_noises) / signal_rates

        noise_loss = jnp.mean((noises - pred_noises) ** 2)
        image_loss = jnp.mean((batch - pred_images) ** 2)
        return noise_loss, image_loss

    # ---- Training loop ----
    history = {"epoch": [], "train_noise_loss": [], "train_image_loss": [],
               "val_noise_loss": [], "val_image_loss": [], "time_s": []}
    rng = jax.random.PRNGKey(42)
    batch_size = args.batch_size

    n_train = train_data.shape[0]
    n_val = val_data.shape[0]
    print(f"\nStarting training for {args.epochs} epochs (from epoch {start_epoch + 1})")
    print(f"  Train samples: {n_train}, Val samples: {n_val}")
    print(f"  Batch size: {batch_size}, LR: {args.lr}, WD: {args.weight_decay}")
    print(f"  EMA: {args.ema_val}")
    print()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()

        # --- Shuffle indices (not the data) ---
        rng, shuffle_rng = jax.random.split(rng)
        perm = np.array(jax.random.permutation(shuffle_rng, n_train))

        n_batches = n_train // batch_size
        epoch_noise_loss = 0.0
        epoch_image_loss = 0.0

        log_every = 100
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            batch = jnp.array(get_batch(train_data, idx))
            rng, step_rng = jax.random.split(rng)

            trainable_vars, non_trainable_vars, opt_state, n_loss, i_loss = train_step(
                trainable_vars, non_trainable_vars, opt_state, batch, step_rng)

            # EMA update — must use ALL network weights in correct order
            all_network_vars = _merge_vars(trainable_vars, non_trainable_vars)
            ema_vars = ema_update(ema_vars, all_network_vars, args.ema_val)

            epoch_noise_loss += float(n_loss)
            epoch_image_loss += float(i_loss)

            if (b + 1) % log_every == 0 or b == 0:
                running_noise = epoch_noise_loss / (b + 1)
                running_image = epoch_image_loss / (b + 1)
                print(f"  [{b+1:6d}/{n_batches}] "
                      f"n_loss={float(n_loss):.5f} i_loss={float(i_loss):.5f} "
                      f"(avg: {running_noise:.5f} / {running_image:.5f})",
                      flush=True)

        avg_noise = epoch_noise_loss / max(n_batches, 1)
        avg_image = epoch_image_loss / max(n_batches, 1)

        # --- Validation ---
        val_noise_total = 0.0
        val_image_total = 0.0
        n_val_batches = max(1, n_val // batch_size)
        for b in range(n_val_batches):
            val_idx = np.arange(b * batch_size, min((b + 1) * batch_size, n_val))
            vbatch = jnp.array(get_batch(val_data, val_idx))
            rng, val_rng = jax.random.split(rng)
            vn, vi = val_step(ema_vars, vbatch, val_rng)
            val_noise_total += float(vn)
            val_image_total += float(vi)
        val_noise = val_noise_total / n_val_batches
        val_image = val_image_total / n_val_batches

        elapsed = time.time() - t0
        history["epoch"].append(epoch)
        history["train_noise_loss"].append(avg_noise)
        history["train_image_loss"].append(avg_image)
        history["val_noise_loss"].append(val_noise)
        history["val_image_loss"].append(val_image)
        history["time_s"].append(elapsed)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train n_loss={avg_noise:.5f} i_loss={avg_image:.5f} | "
              f"val n_loss={val_noise:.5f} i_loss={val_image:.5f} | "
              f"{elapsed:.1f}s")

        # --- Sync weights back to Keras model (for sampling / saving) ---
        for w, v in zip(model.network.trainable_weights, trainable_vars):
            w.assign(v)
        for w, v in zip(model.network.non_trainable_weights, non_trainable_vars):
            w.assign(v)
        for w, v in zip(model.ema_network.weights, ema_vars):
            w.assign(v)

        # --- Checkpoint ---
        if epoch % args.checkpoint_every == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch:03d}")
            save_training_state(ckpt_dir, trainable_vars, non_trainable_vars,
                                ema_vars, opt_state, epoch)
            preset_dir = os.path.join(ckpt_dir, "model_preset")
            save_for_inference(model, preset_dir)
            print(f"  Saved checkpoint to {ckpt_dir}")

        # --- Sample generation ---
        if epoch % args.sample_every == 0:
            generate_and_save_samples(model, epoch, args.output_dir, args.n_sample_steps)

    # ---- Final save ----
    final_dir = os.path.join(args.output_dir, "final_model")
    save_for_inference(model, final_dir)
    save_training_state(
        os.path.join(args.output_dir, "final_state"),
        trainable_vars, non_trainable_vars, ema_vars, opt_state, args.epochs)
    print(f"\nFinal model saved to {final_dir}")

    # ---- Save history ----
    hist_path = os.path.join(args.output_dir, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {hist_path}")

    # ---- Plot loss curves ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = history["epoch"]
    ax1.plot(epochs, history["train_noise_loss"], label="train")
    ax1.plot(epochs, history["val_noise_loss"], label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Noise Loss (MSE)")
    ax1.set_title("Noise Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_image_loss"], label="train")
    ax2.plot(epochs, history["val_image_loss"], label="val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Image Loss (MSE)")
    ax2.set_title("Image Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Training Loss Curves")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "loss_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Loss curves saved.")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()

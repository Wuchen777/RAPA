# RAPA

Code for **RAPA: Relational-Aware Adversarial Patch Attack for Vision-Language-Action
Models**.

RAPA moves the attack target from the visual feature space to the relation space
between visual tokens. Every visual token is described by a local distribution over
its neighbours, and the perturbation is optimized to break that structure, so the
internal association inside the visual representation is disrupted instead of single
token features alone.

## Method

For each visual token a local relation distribution is built from a
twelve-neighbourhood, eight adjacent positions plus four extended axial positions,
with out-of-bound neighbours masked out. The Jensen-Shannon divergence between the
clean and the adversarial distribution is maximized.

Paired views apply the same random transformation to the patch and to the image,
which stabilizes the direction in which the relation pattern changes. For OpenVLA and
OpenVLA-OFT a cross-branch term keeps the DINOv2 and the SigLIP encoder aligned;
Pi0 has a single image encoder, so that term is dropped for it.

The gradients of the individual objectives are weighted by how well their directions
agree with each other, so an objective that conflicts with the others contributes
less, and the aggregated direction drives the patch update.

## Setup

Clone or install the upstream projects at the repository root before running the
model code:

- `LIBERO/`
- `openvla/`
- `openvla_oft/`
- `openpi/`
- the modified LIBERO RLDS datasets under `dataset/modified_libero_rlds/`
- the model checkpoints each runner expects

Python packages used here are in `requirements.txt`. Each upstream model was run in
its own environment during the experiments, so the exact versions should follow
whatever that model requires.

The runner and the evaluation code use Linux file locking and expect a CUDA host.

## Generating a patch

Patch generation for OpenVLA:

```bash
python results/scheme_one_v3_full_50k/code/generate_patch_openvla.py \
  --vla_path openvla/openvla-7b-finetuned-libero-spatial \
  --data_root_dir dataset/modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --batch_size 16 \
  --max_steps 50000 \
  --save_path outputs/openvla/spatial/seed_7/primary \
  --camera_view primary \
  --seed 7
```

The other two entry points follow the same pattern for OpenVLA-OFT and Pi0. Use a
small `--max_steps` for a smoke test; a full run is expensive.

## Evaluation

The LIBERO evaluation scripts for the three models are under `eval/simulation/Libero/`.
Each one takes the patch produced above together with the corresponding checkpoint.

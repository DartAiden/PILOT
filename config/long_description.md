# PILOT

PILOT is an equivariant diffusion model for pocket-conditioned de novo ligand
generation. This TorchServe integration uses the included `last.pt` CrossDocked
checkpoint through the upstream dataset-free inference path; training data is
not needed at inference time.

## Input

Provide exactly one protein PDB and one reference-ligand SDF. Their coordinates
must use the same reference frame. The protein should include the residues that
form the binding pocket. When a matching pocket-residue `.txt` file is absent,
the handler selects standard protein residues within the configured distance
cutoff of the reference ligand.

An optional JSON file may set:

```json
{
  "num_samples": 10,
  "batch_size": 10,
  "max_sample_iter": 20,
  "pocket_cutoff": 7.0,
  "filter_by_lipinski": false
}
```

`num_samples` is the target number of candidates to retain (1–100), `batch_size`
is the number generated per sampling batch (1–40), `max_sample_iter` is the
maximum number of sampling rounds (1–50), and `pocket_cutoff` is measured in
angstroms (1–12). Lipinski filtering is optional.

## Output

The result ZIP contains an SDF of generated structures and a JSON summary with
the requested and generated candidate counts and sampling settings. These are
model-generated candidates, not affinity predictions, docking scores, or
experimental results.

## Upstream

- [PILOT paper](https://doi.org/10.1039/D4SC03523B)

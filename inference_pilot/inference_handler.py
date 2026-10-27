import json
import os
import re
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Tuple

import torch
from rdkit import Chem
from ts.handler_utils.timer import timed

from qbio.file_handler import FileHandler
from qbio.inference_handler import InferenceHandler
from qbio.types import OutputTypes

from inference_pilot.inference_types import InferenceInput, PilotConfig


class PilotHandler(InferenceHandler):
    def __init__(self):
        super().__init__()
        self.source_root = None
        self.out_dir = None

    def initialize(self, context):
        super().initialize(context)
        source_archive = os.path.join(self.model_dir, "pilot_source.tar.gz")
        self.source_root = os.path.join(self.model_dir, "pilot_source")
        os.makedirs(self.source_root, exist_ok=True)
        with tarfile.open(source_archive, "r:gz") as archive:
            archive.extractall(self.source_root)
        if self.source_root not in sys.path:
            sys.path.insert(0, self.source_root)
        if not os.path.isfile(os.path.join(self.model_dir, "last.pt")):
            raise RuntimeError("The PILOT checkpoint last.pt is missing.")

    def _read_config(self, config_path: Path) -> PilotConfig:
        try:
            with config_path.open("r", encoding="utf-8") as file:
                return PilotConfig.model_validate(json.load(file))
        except json.JSONDecodeError as error:
            raise ValueError("Sampling configuration must be valid JSON.") from error

    @timed
    def preprocess(
        self, data
    ) -> Tuple[Path, Path, PilotConfig, InferenceInput]:
        model_input = super().validate_input(data, InferenceInput)
        if not re.fullmatch(r"[A-Za-z0-9._-]+", model_input.id):
            raise ValueError("Request id contains unsupported filename characters.")
        files = FileHandler.get_inference_input_files(
            model_input, self.env_config.LOCAL_EXECUTION, [".pdb", ".sdf", ".json"]
        )
        pdb_files = files.get(".pdb", [])
        sdf_files = files.get(".sdf", [])
        config_files = files.get(".json", [])
        if len(pdb_files) != 1 or len(sdf_files) != 1:
            raise ValueError("Provide exactly one protein PDB and one reference-ligand SDF.")
        if len(config_files) > 1:
            raise ValueError("Provide at most one JSON sampling configuration.")

        config = self._read_config(Path(config_files[0])) if config_files else PilotConfig()
        self.out_dir = os.path.join(
            os.path.dirname(os.path.dirname(str(pdb_files[0]))),
            model_input.output_folder,
        )
        os.makedirs(self.out_dir, exist_ok=True)
        request_dir = Path(self.out_dir, f"pilot_{model_input.id}")
        if request_dir.exists():
            shutil.rmtree(request_dir)
        input_dir = request_dir / "input"
        input_dir.mkdir(parents=True)

        # Upstream derives the protein filename from the reference SDF stem.
        pdb_path = input_dir / "input.pdb"
        sdf_path = input_dir / "input_ref.sdf"
        shutil.copyfile(pdb_files[0], pdb_path)
        shutil.copyfile(sdf_files[0], sdf_path)
        return request_dir, input_dir, config, model_input

    @timed
    def inference(self, data, *args, **kwargs):
        request_dir, input_dir, config, model_input = data
        from experiments.generate_ligands_multi import evaluate, get_args

        result_dir = request_dir / "results"
        arguments = get_args(
            [
                "--model-path",
                os.path.join(self.model_dir, "last.pt"),
                "--dataset-free",
                "--mp-index",
                "1",
                "--num-gpus",
                "1",
                "--test-dir",
                str(input_dir),
                "--save-dir",
                str(result_dir),
                "--num-ligands-per-pocket-to-sample",
                str(config.num_samples),
                "--num-ligands-per-pocket-to-save",
                str(config.num_samples),
                "--batch-size",
                str(config.batch_size),
                "--max-sample-iter",
                str(config.max_sample_iter),
                "--pocket-cutoff",
                str(config.pocket_cutoff),
                "--omit-posebusters",
                "--omit-posecheck",
            ]
        )
        if config.filter_by_lipinski:
            arguments.filter_by_lipinski = True

        evaluate(arguments)
        generated_sdf = result_dir / "sampled" / "input_ref_gen.sdf"
        if not generated_sdf.is_file():
            raise RuntimeError("PILOT could not generate any valid ligand candidates.")
        return generated_sdf, config, model_input

    @timed
    def postprocess(self, data):
        generated_sdf, config, model_input = data
        request_id = model_input.id
        result_sdf = os.path.join(self.out_dir, f"pilot_{request_id}.sdf")
        smiles_path = os.path.join(self.out_dir, f"pilot_{request_id}.smi")
        summary_path = os.path.join(
            self.out_dir, f"pilot_summary_{request_id}.json"
        )
        archive_path = os.path.join(
            self.out_dir, f"pilot_results_{request_id}.zip"
        )
        shutil.copyfile(generated_sdf, result_sdf)

        molecules = [
            molecule
            for molecule in Chem.SDMolSupplier(result_sdf, removeHs=False)
            if molecule is not None
        ]
        smiles = [Chem.MolToSmiles(molecule, canonical=True) for molecule in molecules]
        with open(smiles_path, "w", encoding="utf-8") as file:
            file.writelines(f"{smile}\n" for smile in smiles)
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "model": "pilot",
                    "requested_samples": config.num_samples,
                    "generated_samples": len(molecules),
                    "unique_generated_samples": len(set(smiles)),
                    "sampling_parameters": config.model_dump(),
                    "interpretation": (
                        "Generated ligand candidates only; these results are not "
                        "affinity predictions, docking scores, or experimental findings."
                    ),
                },
                file,
                indent=2,
            )

        FileHandler.archive_files(
            [result_sdf, smiles_path, summary_path], archive_path, self.out_dir
        )
        output_location = FileHandler.upload_files_to_bucket_folder(
            model_input, archive_path, self.env_config.LOCAL_EXECUTION
        )
        summary_location = FileHandler.upload_files_to_bucket_folder(
            model_input, summary_path, self.env_config.LOCAL_EXECUTION
        )
        return super().prepare_inference_response(
            model_input,
            "PILOT",
            {
                OutputTypes.OUTPUT.value: output_location,
                OutputTypes.OUTPUT_SUMMARY.value: summary_location,
            },
        )

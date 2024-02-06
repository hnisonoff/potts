from pathlib import Path
import pandas as pd
import numpy as np
import torch


def get_proteingym_data(dataset):
    alphabet = '-ACDEFGHIKLMNPQRSTVWY'
    aa_to_i = {aa: i for i, aa in enumerate(alphabet)}

    msa_files = list(
        Path('/home/hunter/projects/recombination/ProteinGym/MSA_files').glob(
            "*.a2m"))
    mutation_files = list(
        Path(
            '/home/hunter/projects/recombination/ProteinGym/ProteinGym_substitutions'
        ).glob("*.csv"))
    weight_files = list(
        Path(
            '/home/hunter/projects/recombination/ProteinGym/substitutions_MSAs_all_positions'
        ).glob("*.npy"))
    res_files = list(
        Path('/home/hunter/projects/recombination/ProteinGym/substitutions/').
        glob("*.csv"))

    msa_matches = [f for f in msa_files if dataset.split("_")[0] in f.name]
    if len(msa_matches) > 1:
        msa_matches = [
            f for f in msa_files if "_".join(dataset.split("_")[:2]) in f.name
        ]
    if dataset == 'A4_HUMAN_Seuma_2021':
        msa_matches = [
            Path(
                '/home/hunter/projects/recombination/ProteinGym/MSA_files/A4_HUMAN_full_11-26-2021_b01.a2m'
            )
        ]
    if "P53" in dataset:
        msa_matches = [
            Path(
                '/home/hunter/projects/recombination/ProteinGym/MSA_files/P53_HUMAN_full_04-29-2022_b09.a2m'
            )
        ]

    assert len(msa_matches) == 1
    mut_matches = [f for f in mutation_files if dataset in f.name]
    assert len(mut_matches) == 1
    weight_matches = [
        f for f in weight_files if dataset.split("_")[0] in f.name
    ]
    weight_matches = [
        f for f in weight_files if dataset.split("_")[0] in f.name
    ]
    if len(weight_matches) > 1:
        weight_matches = [
            f for f in weight_files
            if "_".join(dataset.split("_")[:2]) in f.name
        ]
    if dataset == 'A4_HUMAN_Seuma_2021':
        weight_matches = [
            Path(
                '/home/hunter/projects/recombination/ProteinGym/substitutions_MSAs_all_positions/A4_HUMAN_theta_0.2.npy'
            )
        ]
    if "P53" in dataset:
        weight_matches = [
            Path(
                '/home/hunter/projects/recombination/ProteinGym/substitutions_MSAs_all_positions/P53_HUMAN_theta_0.2.npy'
            )
        ]
    assert len(weight_matches) == 1
    res_matches = [f for f in res_files if dataset == f.stem]
    assert len(res_matches) == 1

    msa_fn = msa_matches[0]
    mut_fn = mut_matches[0]
    weight_fn = weight_matches[0]
    res_fn = res_matches[0]
    mut_df = pd.read_csv(mut_fn)
    res_df = pd.read_csv(res_fn)

    y_dms = res_df.DMS_score.to_numpy()
    if dataset == 'SCN5A_HUMAN_Glazer_2019':
        mut_seqs = mut_df.mutated_sequence.map(
            lambda x: [aa_to_i[x[i]] for i in range(len(x))]).to_list()
        mut_seqs = np.asarray(mut_seqs)
        mut_seqs = mut_seqs[:, 1610:1642]
    elif dataset == 'POLG_HCVJF_Qi_2014':
        mut_seqs = mut_df.mutated_sequence.map(
            lambda x: [aa_to_i[x[i]] for i in range(len(x))]).to_list()
        mut_seqs = np.asarray(mut_seqs)
        mut_seqs = mut_seqs[:, 1983:2089]
    elif dataset == 'POLG_CXB3N_Mattenberger_2021':
        mut_seqs = mut_df.mutated_sequence.map(
            lambda x: [aa_to_i[x[i]] for i in range(len(x))]).to_list()
        mut_seqs = np.asarray(mut_seqs)
        mut_seqs = mut_seqs[:, :861]
    elif dataset == 'KCNH2_HUMAN_Kozek_2020':
        mut_seqs = mut_df.mutated_sequence.map(
            lambda x: [aa_to_i[x[i]] for i in range(len(x))]).to_list()
        mut_seqs = np.asarray(mut_seqs)
        mut_seqs = mut_seqs[:, 534:534 + 31]
    elif dataset == 'A0A140D2T1_ZIKV_Sourisseau_growth_2019':
        mut_seqs = mut_df.mutated_sequence.map(
            lambda x: [aa_to_i[x[i]] for i in range(len(x))]).to_list()
        mut_seqs = np.asarray(mut_seqs)
        mut_seqs = mut_seqs[:, 280:804]
    else:
        mut_seqs = mut_df.mutated_sequence.map(
            lambda x: [aa_to_i[x[i]] for i in range(len(x))]).to_list()

    mut_seqs = torch.tensor(mut_seqs)
    res_df = pd.read_csv(res_fn)
    weights = np.load(weight_fn)
    return msa_fn, weights, res_df, mut_seqs

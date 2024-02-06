import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Bio import PDB, SeqIO
import esm
from esm.inverse_folding.util import CoordBatchConverter
import torch
import torch.nn.functional as F
from .potts import Potts
from esm import pretrained

ESMINV_MODEL, ALPHABET = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
ALPHABET_PROTEIN = '-ACDEFGHIKLMNPQRSTVWY'
aa_to_i = {aa: i for i, aa in enumerate(ALPHABET_PROTEIN)}
i_to_a = {i: aa for i, aa in enumerate(ALPHABET_PROTEIN)}


def potts_from_nlls(mutants, nlls, contact_map, native_seq):
    count = 1
    seq_len = len(mutants[0])
    num_tokens = 21
    model = Potts(seq_len, num_tokens)
    L, A = model.L, model.A
    h = model.h.reshape(L, A).detach().clone()
    W = model.reshape_to_L_L_A_A().detach().clone()
    W *= 0.0
    # Generate all single mutants
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    native_seq = list(native_seq)

    wt_nll = nlls[0]
    for i in range(len(native_seq)):
        wt_aa = native_seq[i]
        for aa in amino_acids:
            if aa == wt_aa:
                continue
            seq = native_seq.copy()
            seq[i] = aa

            nll = nlls[count] - wt_nll
            h[i, aa_to_i[aa]] = torch.tensor(nll)
            count += 1

    # Generate all double mutants at contacting positions
    i_idx, j_idx = np.nonzero(contact_map)
    mask = i_idx < j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    for i, j in zip(i_idx, j_idx):
        wt_aa_i = native_seq[i]
        wt_aa_j = native_seq[j]
        for aa_i in amino_acids:
            for aa_j in amino_acids:
                if aa_i == wt_aa_i or aa_j == wt_aa_j:
                    continue
                seq = native_seq.copy()
                seq[i] = aa_i
                seq[j] = aa_j
                nll = nlls[count] - wt_nll
                nll -= h[i, aa_to_i[aa_i]]
                nll -= h[j, aa_to_i[aa_j]]
                W[i, j, aa_to_i[aa_i], aa_to_i[aa_j]] = nll.clone().detach()
                W[j, i, aa_to_i[aa_j], aa_to_i[aa_i]] = nll.clone().detach()
                count += 1
    model = Potts(h=h, W=W.transpose(1, 2).reshape(L * A, L * A))
    return model


def esm_inv_batch_encoding(fpath, bs=30, chain_id="A"):
    model = ESMINV_MODEL.eval().cuda()
    structure = esm.inverse_folding.util.load_structure(fpath, chain_id)
    coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(
        structure)
    seqs = [native_seq for _ in range(bs)]
    with torch.no_grad():
        batch_converter = CoordBatchConverter(ALPHABET)
        batch = [(coords, None, None) for seq in seqs]
        crds, confidence, strs, tokens, padding_mask = batch_converter(
            batch, device='cuda')
        encoder_out = model.encoder(crds, padding_mask, confidence)
    model.cpu()
    return encoder_out, coords, native_seq


def esm_inv_nll_from_encoder(
    encoder_out,
    coords,
    seqs,
):
    model = ESMINV_MODEL.eval().cuda()
    batch_converter = CoordBatchConverter(ALPHABET)
    with torch.no_grad():
        encoder_bs = encoder_out['encoder_out'][0].shape[1]
        if encoder_bs != len(seqs):
            batch_converter = CoordBatchConverter(ALPHABET)
            batch = [(coords, None, None) for seq in seqs]
            crds, confidence, strs, tokens, padding_mask = batch_converter(
                batch, device='cuda')
            encoder_out = model.encoder(crds, padding_mask, confidence)
        batch = [(coords, None, seq) for seq in seqs]
        crds, confidence, strs, tokens, padding_mask = batch_converter(
            batch, device='cuda')
        prev_output_tokens = tokens[:, :-1]
        target = tokens[:, 1:]
        logits, extra = model.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=False,
            return_all_hiddens=False,
        )
        loss = F.cross_entropy(logits, target, reduction='none')
        model.cpu()
        return (np.sum(loss.cpu().numpy(), axis=1))


def get_single_double_mutants(fpath, dist_thresh=6.0):
    dist_map, native_seq = parse_pdb(fpath)
    contact_map = binarize_contact_map(dist_map, dist_thresh)
    print('Native sequence (length {}):'.format(len(native_seq)))
    print(native_seq)
    print('{} structural contacts with threshold={}A'.format(
        0.5 * (np.sum(contact_map) - contact_map.shape[0]), dist_thresh))
    assert len(native_seq) == dist_map.shape[0]

    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    mutants = [native_seq]
    native_seq = list(native_seq)

    # Generate all single mutants
    for i in range(len(native_seq)):
        wt_aa = native_seq[i]
        for aa in amino_acids:
            if aa == wt_aa:
                continue
            seq = native_seq.copy()
            seq[i] = aa
            mutants.append(''.join(seq))

    # Generate all double mutants at contacting positions
    i_idx, j_idx = np.nonzero(contact_map)
    mask = i_idx < j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    for i, j in zip(i_idx, j_idx):
        wt_aa_i = native_seq[i]
        wt_aa_j = native_seq[j]
        for aa_i in amino_acids:
            for aa_j in amino_acids:
                if aa_i == wt_aa_i or aa_j == wt_aa_j:
                    continue
                seq = native_seq.copy()
                seq[i] = aa_i
                seq[j] = aa_j
                mutants.append(''.join(seq))
    return mutants, contact_map, native_seq


def parse_pdb(file_name,
              chain_id=None,
              distance_type='minimum',
              missing_value=-1.):
    print('Loading distance map from PDB: {}'.format(file_name))
    structure_name = os.path.splitext(os.path.basename(file_name))[0]
    structure = PDB.PDBParser().get_structure(structure_name, file_name)
    chain_ids = [chain.id for chain in structure[0]]
    chain_id = chain_ids[0] if chain_id is None else chain_id
    try:
        distance_map = structure_to_contact_map(structure, chain_id,
                                                missing_value, distance_type)
    except KeyError as e:
        print(
            'Distance map for {} could not be loaded due to the following KeyError: {}'
            .format(structure_name, e))
        return None
    print('{} of {} missing in PDB'.format(
        np.sum(np.diag(distance_map) == missing_value), distance_map.shape[0]))
    seq = None
    for record in SeqIO.parse(file_name, 'pdb-atom'):
        if record.annotations['chain'] == chain_id:
            seq = record.seq
    # Get inter-chain contact.
    if len(chain_ids) > 1:
        for chain_id in chain_ids[1:]:
            try:
                dmap = get_interchain_contact_map(structure, chain_ids[0],
                                                  chain_id, missing_value,
                                                  distance_type)
            except KeyError as e:
                print(
                    'Distance map for {} (chains {} and {}) could not be loaded due to the following KeyError: {}'
                    .format(structure_name, chain_ids[0], chain_id, e))
                return None
            distance_map = np.minimum(distance_map, dmap)
    return distance_map, str(seq)


def compute_mean_pairwise_distance(ind, contact_map):
    contact_map = contact_map[ind][:, ind]
    return np.sum(contact_map) / max(1,
                                     contact_map.size - contact_map.shape[0])


def print_contact_map_stats(contact_map):
    cm_vec = contact_map_to_vector(contact_map)
    print('Contact Map Summary Statistics:')
    print('\tMin = {:.4f}'.format(np.amin(cm_vec)))
    print('\tMean +/- std = {:.4f} +/- {:.4f}'.format(np.mean(cm_vec),
                                                      np.std(cm_vec)))
    print('\tMedian = {:.4f}'.format(np.median(cm_vec)))
    print('\tMax = {:.4f}'.format(np.amax(cm_vec)))


def histogram_contact_map(contact_map, figsize=(6, 4)):
    f, ax = plt.subplots(1, 1, figsize=figsize)
    cm_vec = contact_map_to_vector(contact_map)
    sns.distplot(cm_vec, axlabel='C-alpha Distance', ax=ax)
    ax.axvline(np.mean(cm_vec), linestyle='--', color='k', label='mean')
    ax.axvline(np.median(cm_vec), linestyle='--', color='m', label='median')
    ax.legend()


def plot_contact_map(contact_map, figsize=(10, 10)):
    f, axes = plt.subplots(1, 1, figsize=figsize, squeeze=False)
    sns.heatmap(contact_map, cmap=sns.cm.rocket_r)


def calculate_c_alpha_distance(res_x, res_y):
    """Returns the C-alpha distance between two residues."""
    return np.linalg.norm(res_x['CA'].coord - res_y['CA'].coord)


def calculate_minimum_distance(res_x, res_y):
    """Returns the minimum distance between any two atoms in two residues."""
    md = -1
    for atom1 in res_x:
        for atom2 in res_y:
            dist = atom1 - atom2  # biopython defines this operator.
            if md == -1 or dist < md:
                md = dist
    return md


def calculate_contact_map(chain,
                          seq_len=None,
                          missing_value=None,
                          distance_type='c_alpha',
                          chain_b=None):
    """Returns a matrix of C-alpha distances between residue pairs."""
    if chain_b is None:
        chain_b = chain
    if seq_len == None:
        seq_len = len(chain)
        seq_len_b = len(chain_b)
    else:
        seq_len_b = seq_len
    result = np.ones((seq_len, seq_len_b)) * -1.0
    for i in range(len(chain)):
        for j in range(len(chain_b)):
            seqind_i, seqind_j = get_residue_index(
                chain[i]), get_residue_index(chain_b[j])
            if distance_type == 'c_alpha':
                result[seqind_i, seqind_j] = calculate_c_alpha_distance(
                    chain[i], chain_b[j])
            elif distance_type == 'minimum':
                result[seqind_i, seqind_j] = calculate_minimum_distance(
                    chain[i], chain_b[j])
    if missing_value == None:
        # Set missing distances larger than maximum distance in structure.
        missing_value = np.amax(result) + 1.0
    if missing_value == 'median':
        missing_value = np.median(result)
    result[np.less(result, 0)] = missing_value
    return result


def binarize_contact_map(contact_map, threshold=8.0):
    """Returns binary version of contact map."""
    return np.less(contact_map, threshold)


def contact_map_to_vector(contact_map):
    ind = np.triu_indices(contact_map.shape[0], k=1)
    return contact_map[ind]


def get_number_contacts(contact_map, threshold=8.0):
    """Returns the number of residue pairs in contact."""
    contact_vector = contact_map_to_vector(contact_map)
    return np.sum(np.less(contact_vector, threshold))


def get_chain_length(residues, missing_residues=None):
    missing_ind = 0
    if missing_residues != None and len(missing_residues):
        missing_ind = max([residue['ssseq'] for residue in missing_residues])
    residue_ind = get_residue_index(residues[-1])
    return max(residue_ind + 1, missing_ind)


def get_residue_index(residue):
    """Returns the index of the residue in the chain's full sequence."""
    # Adjust by 1 for 0 indexing
    return residue.id[1] - 1


def get_residue_chain(pdb_structure, chain_id):
    """Converts from PDB chain object to list of amino acid residues."""
    missing_residues = None
    if pdb_structure.header['has_missing_residues']:
        missing_residues = [
            r for r in pdb_structure.header['missing_residues']
            if r['chain'] == chain_id
        ]
    chain = pdb_structure[0][chain_id]
    return [
        residue for residue in PDB.Selection.unfold_entities(chain, 'R')
        if residue.id[0] == ' '
    ], missing_residues


def structure_to_contact_map(pdb_structure,
                             chain,
                             missing_value=None,
                             distance_type='c_alpha'):
    """Returns contact map corresponding to given PDB structure object."""
    residues, missing_residues = get_residue_chain(pdb_structure, chain)
    seq_len = get_chain_length(residues, missing_residues)
    return calculate_contact_map(residues, seq_len, missing_value,
                                 distance_type)


def get_interchain_contact_map(pdb_structure,
                               chain_a,
                               chain_b,
                               missing_value=None,
                               distance_type='c_alpha'):
    """Returns distance map corresponding to given chains in PDB structure object."""
    residues_a, missing_residues_a = get_residue_chain(pdb_structure, chain_a)
    residues_b, missing_residues_b = get_residue_chain(pdb_structure, chain_b)
    seq_len = get_chain_length(residues_a, missing_residues_a)
    return calculate_contact_map(residues_a, seq_len, missing_value,
                                 distance_type, residues_b)

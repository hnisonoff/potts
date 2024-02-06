import esm
from esm.inverse_folding.util import CoordBatchConverter
import numpy as np
import torch
import torch.nn.functional as F

ESMINV_MODEL, ALPHABET = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
ALPHABET_PROTEIN = '-ACDEFGHIKLMNPQRSTVWY'
aa_to_i = {aa: i for i, aa in enumerate(ALPHABET_PROTEIN)}
i_to_a = {i: aa for i, aa in enumerate(ALPHABET_PROTEIN)}


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

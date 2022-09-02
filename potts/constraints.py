def map_wt_idx_to_potts_idx(wt_seq):
    potts_idx = 0
    wt_idx_to_potts_idx = {}
    for i, aa in enumerate(wt_seq):
        if aa.isupper():
            wt_idx_to_potts_idx[i] = potts_idx
            potts_idx += 1
        else:
            wt_idx_to_potts_idx[i] = None
    return wt_idx_to_potts_idx


def get_petase_constrained_residue_info(rl, aa_to_i):
    wt_seq = 'mnfprasrlmqaavlgglmavsaaataqtnpyargpnptaasleasagpftvrsftvsRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVaslngtsssPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSsALPIYDSMSrNAKQFLEINGGSHSCANSGNsnQALIGKKGVAWMKRFMdndtrystfacenpnstrvsdfrtancs'

    # map the index of the wt sequence positions to the corresponding index of the potts model
    wt_idx_to_potts_idx = map_wt_idx_to_potts_idx(wt_seq)
    potts_idx_to_wt_idx = {
        p: w
        for w, p in wt_idx_to_potts_idx.items() if p is not None
    }

    # get import residues to keep fixed
    active_site_resnums = [160, 206, 237]
    active_site_potts_idxs = [
        wt_idx_to_potts_idx[r - 1] for r in active_site_resnums
        if wt_idx_to_potts_idx[r - 1] is not None
    ]
    print([wt_seq[r - 1] for r in active_site_resnums])

    disulfide_resnums = [203, 239, 273, 289]
    disulfide_potts_idxs = [
        wt_idx_to_potts_idx[r - 1] for r in disulfide_resnums
        if wt_idx_to_potts_idx[r - 1] is not None
    ]
    print([wt_seq[r - 1] for r in disulfide_resnums])

    subsite_I_resnums = [19, 87, 161, 185]
    subsite_I_potts_idxs = [
        wt_idx_to_potts_idx[r - 1] for r in subsite_I_resnums
        if wt_idx_to_potts_idx[r - 1] is not None
    ]
    print([wt_seq[r - 1] for r in subsite_I_resnums])

    subsite_II_resnums = [88, 89, 159, 238, 241]
    subsite_II_potts_idxs = [
        wt_idx_to_potts_idx[r - 1] for r in subsite_II_resnums
        if wt_idx_to_potts_idx[r - 1] is not None
    ]
    print([wt_seq[r - 1] for r in subsite_II_resnums])

    # Get the idx in the potts model to fix and the index of the AA to fix
    i_to_aa = {i: aa for aa, i in aa_to_i.items()}

    resnums_to_fix = active_site_resnums + disulfide_resnums + subsite_I_resnums + subsite_II_resnums
    resnums_to_fix_match = [
        r for r in resnums_to_fix if wt_idx_to_potts_idx[r - 1] is not None
    ]
    aa_to_fix_match = [wt_seq[r - 1] for r in resnums_to_fix_match]
    aa_idx_to_fix_active = [aa_to_i[aa] for aa in aa_to_fix_match]
    potts_idx_to_fix_active = [
        wt_idx_to_potts_idx[r - 1] for r in resnums_to_fix_match
        if wt_idx_to_potts_idx[r - 1] is not None
    ]

    # Get the idxs of the block endpoints to fix
    # NOTE: we do not include first and last residue

    n_blocks = 10
    potts_idx_to_fix_endpoints = []
    for block in range(n_blocks):
        if block == 0:
            pos_keep_fixed = [rl.block_to_pos[block][-1]]
        elif block == n_blocks - 1:
            pos_keep_fixed = [rl.block_to_pos[block][0]]
        else:
            pos_keep_fixed = [
                rl.block_to_pos[block][0], rl.block_to_pos[block][-1]
            ]
        potts_idx_to_fix_endpoints.extend(pos_keep_fixed)

    wt_idx_to_fix_endpoints = [
        potts_idx_to_wt_idx[i] for i in potts_idx_to_fix_endpoints
    ]
    aa_to_fix_endpoints = [wt_seq[i] for i in wt_idx_to_fix_endpoints]
    aa_idx_to_fix_endpoints = [aa_to_i[aa] for aa in aa_to_fix_endpoints]
    # Get all residues to keep fixed and all aa idxs to fix
    all_potts_idx_to_fix = potts_idx_to_fix_active + potts_idx_to_fix_endpoints
    all_aa_idx_to_fix = aa_idx_to_fix_active + aa_idx_to_fix_endpoints
    return all_potts_idx_to_fix, all_aa_idx_to_fix

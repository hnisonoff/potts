import torch


def to_zs_gauge(h, J):
    assert (J.ndim == 4)
    assert (h.ndim == 2)
    L, A = h.shape
    J_zs = (J - J.mean(dim=2).unsqueeze(dim=2) -
            J.mean(dim=3).unsqueeze(dim=3) +
            (J.reshape(L, L, A * A).mean(dim=-1).unsqueeze(2).unsqueeze(3)))
    tmp = (
        J.mean(dim=3).unsqueeze(dim=3) -
        (J.reshape(L, L, A * A).mean(dim=-1).unsqueeze(2).unsqueeze(3))).sum(
            dim=1).squeeze()
    h_zs = (h - h.mean(dim=-1).unsqueeze(1) + tmp)
    return h_zs, J_zs


def to_wt_gauge(h, J, wt_ind):
    '''
    Thank you Akosua!!!!
    '''
    assert (J.ndim == 4)
    assert (h.ndim == 2)
    select_positions = torch.arange(len(wt_ind))
    J_ij_ab = J
    J_ij_ci_b = J[select_positions, :, wt_ind].unsqueeze(2)
    J_ij_a_cj = J.transpose(-1, -2)[:, select_positions, wt_ind].unsqueeze(3)
    J_ij_ci_cj = J[select_positions, :,
                   wt_ind][:, select_positions,
                           wt_ind].unsqueeze(2).unsqueeze(3)
    J_new = J_ij_ab - J_ij_ci_b - J_ij_a_cj + J_ij_ci_cj

    h_i_c = h[select_positions, wt_ind].unsqueeze(1)
    J_j_nequal_i = (
        J.transpose(-1, -2)[:, select_positions, wt_ind].unsqueeze(3) -
        J[select_positions, :, wt_ind][:, select_positions,
                                       wt_ind].unsqueeze(2).unsqueeze(3)).sum(
                                           dim=1).squeeze()
    J_j_equal_i = (
        J.transpose(-1, -2)[:, select_positions, wt_ind].unsqueeze(3) -
        J[select_positions, :, wt_ind][:, select_positions,
                                       wt_ind].unsqueeze(2).unsqueeze(3)
    )[select_positions, select_positions].squeeze()
    J_j_equal_i = 0
    h_new = (h - h_i_c + (J_j_nequal_i - J_j_equal_i)).to(torch.float)
    return h_new, J_new

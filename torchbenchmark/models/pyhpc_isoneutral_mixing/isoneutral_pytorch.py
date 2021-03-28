import torch


def get_drhodT(salt, temp, p):
    rho0 = 1024.0
    z0 = 0.0
    theta0 = 283.0 - 273.15
    grav = 9.81
    betaT = 1.67e-4
    betaTs = 1e-5
    gammas = 1.1e-8

    zz = -p - z0
    thetas = temp - theta0
    return -(betaTs * thetas + betaT * (1 - gammas * grav * zz * rho0)) * rho0


def get_drhodS(salt, temp, p):
    betaS = 0.78e-3
    rho0 = 1024.
    return betaS * rho0 * torch.ones_like(temp)


def dm_taper(sx):
    """
    tapering function for isopycnal slopes
    """
    iso_slopec = 1e-3
    iso_dslope = 1e-3
    return 0.5 * (1. + torch.tanh((-torch.abs(sx) + iso_slopec) / iso_dslope))


def isoneutral_diffusion_pre(device: torch.device, maskT, maskU, maskV, maskW, dxt, dxu, dyt, dyu, dzt, dzw, cost, cosu, salt, temp, zt, K_iso, K_11, K_22, K_33, Ai_ez, Ai_nz, Ai_bx, Ai_by):
    """
    Isopycnal diffusion for tracer
    following functional formulation by Griffies et al
    Code adopted from MOM2.1
    """
    epsln = 1e-20
    K_iso_steep = 50.
    tau = 0

    dTdx = torch.zeros_like(K_11)
    dSdx = torch.zeros_like(K_11)
    dTdy = torch.zeros_like(K_11)
    dSdy = torch.zeros_like(K_11)
    dTdz = torch.zeros_like(K_11)
    dSdz = torch.zeros_like(K_11)

    """
    drho_dt and drho_ds at centers of T cells
    """
    drdT = maskT * get_drhodT(
        salt[:, :, :, tau], temp[:, :, :, tau], torch.abs(zt)
    )
    drdS = maskT * get_drhodS(
        salt[:, :, :, tau], temp[:, :, :, tau], torch.abs(zt)
    )

    """
    gradients at top face of T cells
    """
    dTdz[:, :, :-1] = maskW[:, :, :-1] * \
        (temp[:, :, 1:, tau] - temp[:, :, :-1, tau]) / \
        dzw[None, None, :-1]
    dSdz[:, :, :-1] = maskW[:, :, :-1] * \
        (salt[:, :, 1:, tau] - salt[:, :, :-1, tau]) / \
        dzw[None, None, :-1]

    """
    gradients at eastern face of T cells
    """
    dTdx[:-1, :, :] = maskU[:-1, :, :] * (temp[1:, :, :, tau] - temp[:-1, :, :, tau]) \
        / (dxu[:-1, None, None] * cost[None, :, None])
    dSdx[:-1, :, :] = maskU[:-1, :, :] * (salt[1:, :, :, tau] - salt[:-1, :, :, tau]) \
        / (dxu[:-1, None, None] * cost[None, :, None])

    """
    gradients at northern face of T cells
    """
    dTdy[:, :-1, :] = maskV[:, :-1, :] * \
        (temp[:, 1:, :, tau] - temp[:, :-1, :, tau]) \
        / dyu[None, :-1, None]
    dSdy[:, :-1, :] = maskV[:, :-1, :] * \
        (salt[:, 1:, :, tau] - salt[:, :-1, :, tau]) \
        / dyu[None, :-1, None]

    """
    Compute Ai_ez and K11 on center of east face of T cell.
    """
    diffloc = torch.zeros_like(K_11)
    diffloc[1:-2, 2:-2, 1:] = 0.25 * (K_iso[1:-2, 2:-2, 1:] + K_iso[1:-2, 2:-2, :-1]
                                      + K_iso[2:-1, 2:-2, 1:] + K_iso[2:-1, 2:-2, :-1])
    diffloc[1:-2, 2:-2, 0] = 0.5 * \
        (K_iso[1:-2, 2:-2, 0] + K_iso[2:-1, 2:-2, 0])

    sumz = torch.zeros_like(K_11)[1:-2, 2:-2]
    for kr in range(2):
        ki = 0 if kr == 1 else 1
        if kr == 1:
            su = K_11.shape[2]
        else:
            su = K_11.shape[2] - 1
        for ip in range(2):
            drodxe = drdT[1 + ip:-2 + ip, 2:-2, ki:] * dTdx[1:-2, 2:-2, ki:] \
                + drdS[1 + ip:-2 + ip, 2:-2, ki:] * dSdx[1:-2, 2:-2, ki:]
            drodze = drdT[1 + ip:-2 + ip, 2:-2, ki:] * dTdz[1 + ip:-2 + ip, 2:-2, :su] \
                + drdS[1 + ip:-2 + ip, 2:-2, ki:] * \
                dSdz[1 + ip:-2 + ip, 2:-2, :su]
            sxe = -drodxe / (torch.min(drodze, torch.tensor([0.], device=device)) - epsln)
            taper = dm_taper(sxe)
            sumz[:, :, ki:] += dzw[None, None, :su] * maskU[1:-2, 2:-2, ki:] \
                * torch.max(torch.tensor([K_iso_steep], device=device), diffloc[1:-2, 2:-2, ki:] * taper)
            Ai_ez[1:-2, 2:-2, ki:, ip, kr] = taper * \
                sxe * maskU[1:-2, 2:-2, ki:]
    K_11[1:-2, 2:-2, :] = sumz / (4. * dzt[None, None, :])

    """
    Compute Ai_nz and K_22 on center of north face of T cell.
    """
    diffloc[...] = 0
    diffloc[2:-2, 1:-2, 1:] = 0.25 * (K_iso[2:-2, 1:-2, 1:] + K_iso[2:-2, 1:-2, :-1]
                                      + K_iso[2:-2, 2:-1, 1:] + K_iso[2:-2, 2:-1, :-1])
    diffloc[2:-2, 1:-2, 0] = 0.5 * \
        (K_iso[2:-2, 1:-2, 0] + K_iso[2:-2, 2:-1, 0])

    sumz = torch.zeros_like(K_11)[2:-2, 1:-2]
    for kr in range(2):
        ki = 0 if kr == 1 else 1
        if kr == 1:
            su = K_11.shape[2]
        else:
            su = K_11.shape[2] - 1
        for jp in range(2):
            drodyn = drdT[2:-2, 1 + jp:-2 + jp, ki:] * dTdy[2:-2, 1:-2, ki:] + \
                drdS[2:-2, 1 + jp:-2 + jp, ki:] * dSdy[2:-2, 1:-2, ki:]
            drodzn = drdT[2:-2, 1 + jp:-2 + jp, ki:] * dTdz[2:-2, 1 + jp:-2 + jp, :su] \
                + drdS[2:-2, 1 + jp:-2 + jp, ki:] * \
                dSdz[2:-2, 1 + jp:-2 + jp, :su]
            syn = -drodyn / (torch.min(torch.tensor([0.], device=device), drodzn) - epsln)
            taper = dm_taper(syn)
            sumz[:, :, ki:] += dzw[None, None, :su] \
                * maskV[2:-2, 1:-2, ki:] * torch.max(torch.tensor([K_iso_steep], device=device), diffloc[2:-2, 1:-2, ki:] * taper)
            Ai_nz[2:-2, 1:-2, ki:, jp, kr] = taper * \
                syn * maskV[2:-2, 1:-2, ki:]
    K_22[2:-2, 1:-2, :] = sumz / (4. * dzt[None, None, :])

    """
    compute Ai_bx, Ai_by and K33 on top face of T cell.
    """
    sumx = torch.zeros_like(K_11)[2:-2, 2:-2, :-1]
    sumy = torch.zeros_like(K_11)[2:-2, 2:-2, :-1]

    for kr in range(2):
        if kr == 1:
            sl = 1
            su = K_11.shape[2]
        else:
            sl = 0
            su = K_11.shape[2] - 1

        drodzb = drdT[2:-2, 2:-2, sl:su] * dTdz[2:-2, 2:-2, :-1] \
            + drdS[2:-2, 2:-2, sl:su] * dSdz[2:-2, 2:-2, :-1]

        # eastward slopes at the top of T cells
        for ip in range(2):
            drodxb = drdT[2:-2, 2:-2, sl:su] * dTdx[1 + ip:-3 + ip, 2:-2, sl:su] \
                + drdS[2:-2, 2:-2, sl:su] * \
                dSdx[1 + ip:-3 + ip, 2:-2, sl:su]
            sxb = -drodxb / (torch.min(torch.tensor([0.], device=device), drodzb) - epsln)
            taper = dm_taper(sxb)
            sumx += dxu[1 + ip:-3 + ip, None, None] * \
                K_iso[2:-2, 2:-2, :-1] * taper * \
                sxb**2 * maskW[2:-2, 2:-2, :-1]
            Ai_bx[2:-2, 2:-2, :-1, ip, kr] = taper * \
                sxb * maskW[2:-2, 2:-2, :-1]

        # northward slopes at the top of T cells
        for jp in range(2):
            facty = cosu[1 + jp:-3 + jp] * dyu[1 + jp:-3 + jp]
            drodyb = drdT[2:-2, 2:-2, sl:su] * dTdy[2:-2, 1 + jp:-3 + jp, sl:su] \
                + drdS[2:-2, 2:-2, sl:su] * \
                dSdy[2:-2, 1 + jp:-3 + jp, sl:su]
            syb = -drodyb / (torch.min(torch.tensor([0.], device=device), drodzb) - epsln)
            taper = dm_taper(syb)
            sumy += facty[None, :, None] * K_iso[2:-2, 2:-2, :-1] \
                * taper * syb**2 * maskW[2:-2, 2:-2, :-1]
            Ai_by[2:-2, 2:-2, :-1, jp, kr] = taper * \
                syb * maskW[2:-2, 2:-2, :-1]

    K_33[2:-2, 2:-2, :-1] = sumx / (4 * dxt[2:-2, None, None]) + \
        sumy / (4 * dyt[None, 2:-2, None]
                * cost[None, 2:-2, None])
    K_33[2:-2, 2:-2, -1] = 0.

    return K_11, K_22, K_33, Ai_ez, Ai_nz, Ai_bx, Ai_by

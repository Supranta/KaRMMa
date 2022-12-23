import math
import torch
import healpy as hp
import numpy as np


class Alm2Map(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alms, nside, lmax):
        ctx.alms = alms
        ctx.nside = nside
        ctx.lmax = lmax

        m = torch.tensor(hp.alm2map(alms.numpy(), nside, lmax=lmax))

        return m

    @staticmethod
    def backward(ctx, grad_output):
        nside = ctx.nside
        lmax = ctx.lmax

        _, emm = hp.Alm.getlm(lmax)
        a = torch.ones(len(emm), dtype=torch.double)
        a[emm > 0] = 2

        grad_out_alm = Map2Alm.apply(grad_output, lmax)
        grad_alm = a * hp.nside2npix(nside) / (4 * math.pi) * grad_out_alm

        return grad_alm, None, None


class Map2Alm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, lmax):
        ctx.m = m
        ctx.nside = hp.npix2nside(len(m))
        ctx.lmax = lmax

        alm = torch.tensor(hp.map2alm(m.numpy(), lmax=lmax, use_pixel_weights=True))

        return alm

    @staticmethod
    def backward(ctx, grad_output):
        nside = ctx.nside
        lmax = ctx.lmax

        _, emm = hp.Alm.getlm(lmax)
        a = torch.ones(len(emm), dtype=torch.double)
        a[emm > 0] = 0.5

        grad_out_m = Alm2Map.apply(a * grad_output, nside, lmax)
        grad_m = 4 * math.pi / hp.nside2npix(nside) * grad_out_m

        return grad_m, None, None


class Alm2MapSpin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, elm, blm, nside, lmax):
        ctx.nside = nside
        ctx.lmax = lmax

        inputs = [np.zeros_like(elm.numpy()), elm.numpy(), blm.numpy()]
        _, q, u = hp.alm2map(inputs, nside, lmax=lmax)
        q = torch.tensor(q)
        u = torch.tensor(u)

        return q, u

    @staticmethod
    def backward(ctx, q_grad, u_grad):
        nside = ctx.nside
        lmax = ctx.lmax

        _, emm = hp.Alm.getlm(lmax)
        a = torch.ones(len(emm), dtype=torch.double)
        a[emm > 0] = 2

        elm_grad, blm_grad = Map2AlmSpin.apply(q_grad, u_grad, lmax)
        elm_grad = a * hp.nside2npix(nside) / (4 * math.pi) * elm_grad
        blm_grad = a * hp.nside2npix(nside) / (4 * math.pi) * blm_grad

        return elm_grad, blm_grad, None, None


class Map2AlmSpin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, u, lmax):
        ctx.nside = hp.npix2nside(len(q))
        ctx.lmax = lmax

        inputs = [np.zeros_like(q.numpy()), q.numpy(), u.numpy()]
        _, elm, blm = hp.map2alm(inputs, lmax=lmax, use_pixel_weights=True)
        elm = torch.tensor(elm)
        blm = torch.tensor(blm)

        return elm, blm

    @staticmethod
    def backward(ctx, elm_grad, blm_grad):
        nside = ctx.nside
        lmax = ctx.lmax

        _, emm = hp.Alm.getlm(lmax)
        a = torch.ones(len(emm), dtype=torch.double)
        a[emm > 0] = 0.5

        q_grad, u_grad = Alm2MapSpin.apply(a * elm_grad, a * blm_grad, nside, lmax)
        q_grad = 4 * math.pi / hp.nside2npix(nside) * q_grad
        u_grad = 4 * math.pi / hp.nside2npix(nside) * u_grad

        return q_grad, u_grad, None, None


class UDGrade(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, out_nside):
        ctx.in_nside = hp.npix2nside(len(m))
        ctx.out_nside = out_nside

        ud_m = torch.tensor(hp.ud_grade(m.numpy(), out_nside))

        return ud_m

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.out_nside > ctx.in_nside:
            fac = (ctx.out_nside / ctx.in_nside) ** 2.
        else:
            fac = (ctx.in_nside / ctx.out_nside) ** (-2.)

        grad = UDGrade.apply(grad_out, ctx.in_nside) * fac

        return grad, None


def shear2conv(g1, g2, lmax=None):
    nside = hp.npix2nside(len(g1))

    gelm, _ = Map2AlmSpin.apply(g1, g2, lmax)

    lmax = hp.Alm.getlmax(len(gelm))
    l, m = hp.Alm.getlm(lmax)
    l = torch.tensor(l, dtype=torch.double)

    good_ls = l > 1
    fac = torch.zeros_like(l)
    l = l[good_ls]
    fac[good_ls] = - torch.sqrt(l * (l + 1) / ((l + 2) * (l - 1)))
    kelm = fac * gelm

    k = Alm2Map.apply(kelm, nside, lmax)

    return k


def conv2shear(k, lmax=None):
    nside = hp.npix2nside(len(k))

    kelm = Map2Alm.apply(k, lmax)

    lmax = hp.Alm.getlmax(len(kelm))
    l, m = hp.Alm.getlm(lmax)
    l = torch.tensor(l, dtype=torch.double)

    good_ls = l > 0
    fac = torch.zeros_like(l)
    l = l[good_ls]
    fac[good_ls] = - torch.sqrt((l + 2) * (l - 1) / (l * (l + 1)))
    gelm = fac * kelm

    gblm = torch.zeros_like(kelm)

    g1, g2 = Alm2MapSpin.apply(gelm, gblm, nside, lmax)

    return g1, g2

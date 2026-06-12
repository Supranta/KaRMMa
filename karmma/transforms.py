import math
import torch
import healpy as hp
import numpy as np
import astropy.io.fits as fits
from functools import cache
import time
import urllib.request
from pathlib import Path


@cache
def get_pixel_weights(nside: int) -> np.ndarray:
    """Return HEALPix full pixel weights (matching use_pixel_weights=True)."""
    nside = int(nside)
    nside_str = f"{nside:04d}"
    filename  = f"healpix_full_weights_nside_{nside_str}.fits"
    cache_dir = Path.home() / ".cache" / "karmma" / "full_weights"
    path      = cache_dir / filename

    if not path.exists():
        url = (
            "https://raw.githubusercontent.com/healpy/healpy-data"
            f"/master/full_weights/{filename}"
        )
        print(f"Downloading pixel weights for nside={nside} from healpy-data...")
        cache_dir.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(url, path)
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  Attempt {attempt+1} failed ({e}), retrying...")
                    time.sleep(2)
                else:
                    path.unlink(missing_ok=True)
                    raise
        print("Download complete.")

    with fits.open(path) as hdul:
        w8list = hdul[1].data.field(0).astype(np.float64)

    npix = hp.nside2npix(nside)
    w8map = np.zeros(npix, dtype=np.float64)

    pnorth = vpix = 0
    for ring in range(2 * nside):
        qpix = min(ring + 1, nside)
        shifted = int(ring < nside - 1 or (ring + nside) % 2 == 1)
        qp4 = 4 * qpix

        for p in range(qp4):
            j4 = p % qpix
            rpix = min(j4, qpix - shifted - j4)
            w8map[pnorth + p] = w8list[vpix + rpix]

        if ring < 2 * nside - 1:
            psouth = npix - pnorth - qp4
            w8map[psouth:psouth + qp4] = w8map[pnorth:pnorth + qp4]

        pnorth += qp4
        vpix += (qpix + 1) // 2 + 1 - ((qpix % 2) | shifted)

    return w8map + 1.0




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

        grad_out_alm = torch.tensor(hp.map2alm(grad_output.numpy(), lmax=lmax, iter=0, use_pixel_weights=False))
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
        n_pix = hp.nside2npix(nside)

        _, emm = hp.Alm.getlm(lmax)
        a = torch.ones(len(emm), dtype=torch.double)
        a[emm > 0] = 0.5

        w = torch.tensor(get_pixel_weights(nside))
        grad_out_m = torch.tensor(hp.alm2map((a * grad_output).numpy(), nside, lmax=lmax))
        grad_m = w * 4 * math.pi / n_pix * grad_out_m

        return grad_m, None



class Alm2MapSpin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, elm, blm, nside, lmax):
        ctx.nside = nside
        ctx.lmax = lmax

        inputs = [np.zeros_like(elm.numpy()), elm.numpy(), blm.numpy()]
        _, q, u = hp.alm2map(inputs, nside, lmax=lmax)

        return torch.tensor(q), torch.tensor(u)

    @staticmethod
    def backward(ctx, q_grad, u_grad):
        nside = ctx.nside
        lmax = ctx.lmax
        n_pix = hp.nside2npix(nside)

        _, emm = hp.Alm.getlm(lmax)
        a = torch.ones(len(emm), dtype=torch.double)
        a[emm > 0] = 2

        inputs = [np.zeros(n_pix, dtype=np.float64), q_grad.numpy(), u_grad.numpy()]
        _, elm_grad, blm_grad = hp.map2alm(inputs, lmax=lmax, iter=0, use_pixel_weights=False)
        elm_grad = a * n_pix / (4 * math.pi) * torch.tensor(elm_grad)
        blm_grad = a * n_pix / (4 * math.pi) * torch.tensor(blm_grad)

        return elm_grad, blm_grad, None, None


class Map2AlmSpin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, u, lmax):
        ctx.nside = hp.npix2nside(len(q))
        ctx.lmax = lmax

        inputs = [np.zeros_like(q.numpy()), q.numpy(), u.numpy()]
        _, elm, blm = hp.map2alm(inputs, lmax=lmax, use_pixel_weights=True)

        return torch.tensor(elm), torch.tensor(blm)

    @staticmethod
    def backward(ctx, elm_grad, blm_grad):
        nside = ctx.nside
        lmax = ctx.lmax
        n_pix = hp.nside2npix(nside)

        _, emm = hp.Alm.getlm(lmax)
        a = torch.ones(len(emm), dtype=torch.double)
        a[emm > 0] = 0.5

        w = torch.tensor(get_pixel_weights(nside))
        inputs = [
            np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128),
            (a * elm_grad).numpy(),
            (a * blm_grad).numpy()
        ]
        _, q_grad, u_grad = hp.alm2map(inputs, nside, lmax=lmax)
        q_grad = w * 4 * math.pi / n_pix * torch.tensor(q_grad)
        u_grad = w * 4 * math.pi / n_pix * torch.tensor(u_grad)

        return q_grad, u_grad, None


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


def conv2shear(k, lmax=None, pixwin=None):
    nside = hp.npix2nside(len(k))

    kelm = Map2Alm.apply(k, lmax)

    if pixwin is not None:
        kelm = kelm * pixwin

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

"""SHT transforms for KaRMMa (PyTorch).

ducc0-based, autograd-compatible. All bins are transformed in a single
ntrans call. nside/lmax/spin are Python ints. Spin-2 is supported
natively via the `spin` argument, so no dummy temperature component is
ever synthesised.

The two `torch.autograd.Function`s here, `_Synthesis` and
`_AdjointSynthesis`, are an exact transpose pair: each one's `backward`
is the other's `apply`. That has two consequences. Every derivative
comes from the ducc0 synthesis/adjoint_synthesis pair rather than a
hand-written approximation, and because the backward pass is itself
built from autograd Functions, higher-order derivatives (Hessians,
Hessian-vector products) work too.

The SHT is real-linear but not complex-linear (complex alm in, real map
out). PyTorch's gradient for a complex tensor `z` under a real loss is
the real gradient packed as ``dL/dRe(z) + i dL/dIm(z)``, so the
transpose is taken in the real (Re, Im) degrees of freedom: the
transpose of `synthesis` is `adjoint_synthesis` with the m>0
coefficients doubled, since each stored m>0 coefficient contributes to
both +m and -m in the real sum. Everything else -- the pixel weights,
the 4 pi / n_pix normalisation, and the compensating 1/2 on m>0 in
`map2alm` -- is plain torch arithmetic and is differentiated natively.

`map2alm` here is exactly one adjoint synthesis with full pixel weights,
so it is the exact transpose of what its `backward` applies. It
reproduces healpy's ``map2alm(..., use_pixel_weights=True)``, whose
Jacobi iterations are skipped on the pixel-weights path.
"""

import time
import urllib.request
from functools import cache, lru_cache
from pathlib import Path

import astropy.io.fits as fits
import ducc0.healpix
import ducc0.sht
import healpy as hp
import numpy as np
import torch

__all__ = [
    "alm2map",
    "map2alm",
    "get_pixel_weights",
    "Alm2Map",
    "Map2Alm",
    "Alm2MapSpin",
    "Map2AlmSpin",
]


@lru_cache(maxsize=1)
def get_pixel_weights(nside: int) -> np.ndarray:
    """Return the HEALPix full pixel weight map for nside, downloaded on first use.

    Equivalent to healpy's use_pixel_weights=True. Results are cached to
    ~/.cache/karmma/full_weights/.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.

    Returns
    -------
    np.ndarray
        Pixel weight map, shape (n_pix,), with values near 1.0 (stored
        as w + 1 per the FITS convention).
    """
    nside = int(nside)
    nside_str = f"{nside:04d}"
    filename = f"healpix_full_weights_nside_{nside_str}.fits"
    cache_dir = Path.home() / ".cache" / "karmma" / "full_weights"
    path = cache_dir / filename

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
                    print(f"  Attempt {attempt + 1} failed ({e}), retrying...")
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
            w8map[psouth : psouth + qp4] = w8map[pnorth : pnorth + qp4]

        pnorth += qp4
        vpix += (qpix + 1) // 2 + 1 - ((qpix % 2) | shifted)

    return w8map + 1.0


@cache
def _geometry(nside: int) -> dict:
    """Return the ducc0 ring description of the RING-ordered HEALPix grid."""
    info = ducc0.healpix.Healpix_Base(nside, "RING").sht_info()
    return {k: info[k] for k in ("theta", "phi0", "nphi", "ringstart")}


@lru_cache(maxsize=4)
def _m_gt0_factor(lmax: int) -> np.ndarray:
    """Return the per-coefficient factor turning adjoint_synthesis into the transpose.

    Parameters
    ----------
    lmax : int
        Maximum multipole.

    Returns
    -------
    np.ndarray
        Shape (n_alm,), 1.0 where m == 0 and 2.0 elsewhere.
    """
    ms = hp.Alm.getlm(lmax)[1]
    return np.where(ms == 0, 1.0, 2.0)


@lru_cache(maxsize=4)
def _map2alm_scale(lmax: int, n_pix: int) -> torch.Tensor:
    """Per-coefficient rescaling that turns the transpose's output into `map2alm`'s.

    Parameters
    ----------
    lmax : int
        Maximum multipole.
    n_pix : int
        Number of map pixels, for the 4 pi / n_pix normalisation.

    Returns
    -------
    torch.Tensor
        Shape (n_alm,), float64. Undoes the transpose's 2x on m>0 and
        applies the 4 pi / n_pix normalisation, so the result matches
        healpy.map2alm(..., use_pixel_weights=True).
    """
    ms = hp.Alm.getlm(lmax)[1]
    scale = np.where(ms == 0, 1.0, 0.5) * (4.0 * np.pi / n_pix)
    return torch.from_numpy(scale)


@lru_cache(maxsize=4)
def _weights_tensor(nside: int) -> torch.Tensor:
    """Return the full pixel weight map for nside as a float64 torch tensor."""
    return torch.from_numpy(get_pixel_weights(nside).copy())


def _to_numpy(x: torch.Tensor, dtype: torch.dtype, name: str) -> np.ndarray:
    """Return a contiguous numpy view of a CPU tensor of the required dtype."""
    if x.device.type != "cpu":
        raise ValueError(
            f"{name} must be a CPU tensor: ducc0 has no GPU backend "
            f"(got device {x.device})"
        )
    if x.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype} (got {x.dtype})")
    return x.detach().contiguous().numpy()


def _as_batched(x: torch.Tensor, spin: int) -> tuple[torch.Tensor, torch.Size]:
    """Reshape to ducc0's (ntrans, ncomp, n) layout, returning the batch shape.

    For spin 0 the last axis is the coefficient/pixel axis and everything
    before it is batch. For spin 2 the last two axes must be
    (2, n) -- the (E, B) or (Q, U) pair -- and everything before them is
    batch.
    """
    if spin == 0:
        return x.reshape(-1, 1, x.shape[-1]), x.shape[:-1]
    if x.ndim < 2 or x.shape[-2] != 2:
        raise ValueError(
            f"spin={spin} requires shape (..., 2, n) for the two components, "
            f"got {tuple(x.shape)}"
        )
    return x.reshape(-1, 2, x.shape[-1]), x.shape[:-2]


def _restore(y: torch.Tensor, batch: torch.Size, spin: int) -> torch.Tensor:
    """Undo `_as_batched`, putting the batch axes back on the ducc0 output."""
    keep = 1 if spin == 0 else 2
    return y.reshape(batch + y.shape[-keep:])


class _Synthesis(torch.autograd.Function):
    """alm -> map, on (ntrans, ncomp, n)-shaped tensors."""

    @staticmethod
    def forward(ctx, alm, nside, lmax, spin, nthreads):
        ctx.cfg = (nside, lmax, spin, nthreads)
        alm_np = _to_numpy(alm, torch.complex128, "alm")
        maps = np.empty(alm_np.shape[:-1] + (hp.nside2npix(nside),), dtype=np.float64)
        ducc0.sht.synthesis(
            alm=alm_np,
            map=maps,
            **_geometry(nside),
            lmax=lmax,
            mmax=lmax,
            spin=spin,
            nthreads=nthreads,
        )
        return torch.from_numpy(maps)

    @staticmethod
    def backward(ctx, grad_maps):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None
        return _AdjointSynthesis.apply(grad_maps, *ctx.cfg), None, None, None, None


class _AdjointSynthesis(torch.autograd.Function):
    """map -> alm, the exact transpose of `_Synthesis`."""

    @staticmethod
    def forward(ctx, maps, nside, lmax, spin, nthreads):
        ctx.cfg = (nside, lmax, spin, nthreads)
        maps_np = _to_numpy(maps, torch.float64, "map")
        alm = np.empty(maps_np.shape[:-1] + (hp.Alm.getsize(lmax),), dtype=np.complex128)
        ducc0.sht.adjoint_synthesis(
            map=maps_np,
            alm=alm,
            **_geometry(nside),
            lmax=lmax,
            mmax=lmax,
            spin=spin,
            nthreads=nthreads,
        )
        alm *= _m_gt0_factor(lmax)
        return torch.from_numpy(alm)

    @staticmethod
    def backward(ctx, grad_alm):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None
        return _Synthesis.apply(grad_alm, *ctx.cfg), None, None, None, None


# -- public API ---------------------------------------------------------------


def alm2map(
    alm: torch.Tensor, nside: int, lmax: int, spin: int = 0, nthreads: int = 0
) -> torch.Tensor:
    """Synthesis SHT via ducc0, batched over any leading axes.

    Parameters
    ----------
    alm : torch.Tensor
        complex128, on CPU. (..., n_alm) for spin=0, (..., 2, n_alm) for
        spin=2, where the pair is (E, B). Leading axes are batch axes and
        are transformed in a single ducc0 call.
    nside : int
        HEALPix resolution.
    lmax : int
        Maximum multipole; also used as mmax.
    spin : int, optional
        0 or 2, by default 0.
    nthreads : int, optional
        ducc0 thread count; 0 (default) means all available hardware
        threads.

    Returns
    -------
    torch.Tensor
        float64. (..., n_pix) for spin=0, (..., 2, n_pix) for spin=2,
        where the pair is (Q, U). Matches healpy's sign convention.
    """
    spin = int(spin)
    alm3, batch = _as_batched(alm, spin)
    maps = _Synthesis.apply(alm3, int(nside), int(lmax), spin, int(nthreads))
    return _restore(maps, batch, spin)


def map2alm(
    maps: torch.Tensor, lmax: int, spin: int = 0, nthreads: int = 0
) -> torch.Tensor:
    """Analysis SHT via ducc0 (adjoint synthesis with full pixel weights).

    Applies full pixel weights and the 4 pi / n_pix normalisation,
    matching healpy.map2alm with use_pixel_weights=True. nside is
    inferred from the map size.

    Parameters
    ----------
    maps : torch.Tensor
        float64, on CPU. (..., n_pix) for spin=0, (..., 2, n_pix) for
        spin=2, where the pair is (Q, U). Leading axes are batch axes and
        are transformed in a single ducc0 call.
    lmax : int
        Maximum multipole; also used as mmax.
    spin : int, optional
        0 or 2, by default 0.
    nthreads : int, optional
        ducc0 thread count; 0 (default) means all available hardware
        threads.

    Returns
    -------
    torch.Tensor
        complex128. (..., n_alm) for spin=0, (..., 2, n_alm) for spin=2,
        where the pair is (E, B).
    """
    spin = int(spin)
    lmax = int(lmax)
    maps3, batch = _as_batched(maps, spin)
    n_pix = maps3.shape[-1]
    nside = hp.npix2nside(n_pix)
    weighted = _weights_tensor(nside) * maps3
    alm = _AdjointSynthesis.apply(weighted, nside, lmax, spin, int(nthreads))
    return _restore(alm * _map2alm_scale(lmax, n_pix), batch, spin)


# -- compatibility shims for the healpy-backed API ----------------------------


class Alm2Map:
    """Compatibility shim: `Alm2Map.apply(alm, nside, lmax)` -> `alm2map`."""

    @staticmethod
    def apply(alms, nside, lmax, nthreads=0):
        return alm2map(alms, nside, lmax, spin=0, nthreads=nthreads)


class Map2Alm:
    """Compatibility shim: `Map2Alm.apply(m, lmax)` -> `map2alm`."""

    @staticmethod
    def apply(m, lmax, nthreads=0):
        return map2alm(m, lmax, spin=0, nthreads=nthreads)


class Alm2MapSpin:
    """Compatibility shim taking and returning the components separately.

    `Alm2MapSpin.apply(elm, blm, nside, lmax)` -> `(q, u)`.
    """

    @staticmethod
    def apply(elm, blm, nside, lmax, nthreads=0):
        qu = alm2map(
            torch.stack([elm, blm], dim=-2), nside, lmax, spin=2, nthreads=nthreads
        )
        return qu[..., 0, :], qu[..., 1, :]


class Map2AlmSpin:
    """Compatibility shim taking and returning the components separately.

    `Map2AlmSpin.apply(q, u, lmax)` -> `(elm, blm)`.
    """

    @staticmethod
    def apply(q, u, lmax, nthreads=0):
        eb = map2alm(torch.stack([q, u], dim=-2), lmax, spin=2, nthreads=nthreads)
        return eb[..., 0, :], eb[..., 1, :]
        
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
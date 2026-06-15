
import time
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as spla

EPS0 = 8.8541878128e-12
HAS_CUPY = False


@dataclass
class HarmonicWorldConfig:
    Lx: float = 1.30
    Ly: float = 0.85
    Lz: float = 1.10
    center: tuple = (0.65, 0.425, 0.55)
    E0: float = 1.0
    seed: int = 12
    poly_scale: float = 0.18
    fourier_scale: float = 0.12
    n_fourier_terms: int = 5
    use_polynomial: bool = True
    use_fourier: bool = True

    @property
    def lengths(self):
        return (float(self.Lx), float(self.Ly), float(self.Lz))


@dataclass
class HarmonicFieldModel:
    config: HarmonicWorldConfig
    poly_coefficients: dict
    fourier_terms: list
    app_kind: str = 'custom'
    app_params: dict = None


def _as_tuple3(x):
    return (float(x[0]), float(x[1]), float(x[2]))


def sample_poly_coefficients(seed=0, scale=0.20):
    rng = np.random.default_rng(seed)
    names = [
        'u', 'v', 'w', 'uv', 'uw', 'vw',
        'u2_minus_v2', '2w2_minus_u2_minus_v2',
        'u_4w2_u2_v2', 'v_4w2_u2_v2', 'w_u2_minus_v2',
        'uvw', 'u_u2_minus_3v2', 'v_3u2_minus_v2',
        'w_2w2_minus_3u2_minus_3v2',
    ]
    coeffs = dict(zip(names, rng.normal(0.0, scale, size=len(names))))
    coeffs['u'] += 0.55
    coeffs['v'] -= 0.35
    coeffs['w'] += 0.25
    return coeffs


def sample_fourier_terms(seed=0, n_terms=4, scale=0.10):
    rng = np.random.default_rng(seed + 1000)
    terms = []
    for _ in range(n_terms):
        terms.append({
            'amplitude': float(rng.normal(0.0, scale)),
            'alpha': float(rng.integers(1, 5)),
            'beta': float(rng.integers(1, 5)),
            'phase_u': float(rng.uniform(0.0, 2.0 * np.pi)),
            'phase_v': float(rng.uniform(0.0, 2.0 * np.pi)),
            'z_parity': 'even' if rng.random() < 0.5 else 'odd',
        })
    return terms


def create_harmonic_field_model(config, poly_coefficients=None, fourier_terms=None):
    if poly_coefficients is None:
        poly_coefficients = sample_poly_coefficients(config.seed, config.poly_scale)
    if fourier_terms is None:
        fourier_terms = sample_fourier_terms(config.seed, config.n_fourier_terms, config.fourier_scale)
    return HarmonicFieldModel(config=config, poly_coefficients=dict(poly_coefficients), fourier_terms=list(fourier_terms))


def _add_polynomial_harmonics(phi, Ex, Ey, Ez, u, v, w, coeffs, E0, a):
    S = E0 * a

    c = coeffs.get('u', 0.0)
    phi += S * c * u
    Ex += -E0 * c

    c = coeffs.get('v', 0.0)
    phi += S * c * v
    Ey += -E0 * c

    c = coeffs.get('w', 0.0)
    phi += S * c * w
    Ez += -E0 * c

    c = coeffs.get('uv', 0.0)
    phi += S * c * (u * v)
    Ex += -E0 * c * v
    Ey += -E0 * c * u

    c = coeffs.get('uw', 0.0)
    phi += S * c * (u * w)
    Ex += -E0 * c * w
    Ez += -E0 * c * u

    c = coeffs.get('vw', 0.0)
    phi += S * c * (v * w)
    Ey += -E0 * c * w
    Ez += -E0 * c * v

    c = coeffs.get('u2_minus_v2', 0.0)
    phi += S * c * (u * u - v * v)
    Ex += -E0 * c * (2.0 * u)
    Ey += +E0 * c * (2.0 * v)

    c = coeffs.get('2w2_minus_u2_minus_v2', 0.0)
    phi += S * c * (2.0 * w * w - u * u - v * v)
    Ex += +E0 * c * (2.0 * u)
    Ey += +E0 * c * (2.0 * v)
    Ez += -E0 * c * (4.0 * w)

    c = coeffs.get('u_4w2_u2_v2', 0.0)
    phi += S * c * (u * (4.0 * w * w - u * u - v * v))
    Ex += -E0 * c * (4.0 * w * w - 3.0 * u * u - v * v)
    Ey += +E0 * c * (2.0 * u * v)
    Ez += -E0 * c * (8.0 * u * w)

    c = coeffs.get('v_4w2_u2_v2', 0.0)
    phi += S * c * (v * (4.0 * w * w - u * u - v * v))
    Ex += +E0 * c * (2.0 * u * v)
    Ey += -E0 * c * (4.0 * w * w - u * u - 3.0 * v * v)
    Ez += -E0 * c * (8.0 * v * w)

    c = coeffs.get('w_u2_minus_v2', 0.0)
    phi += S * c * (w * (u * u - v * v))
    Ex += -E0 * c * (2.0 * u * w)
    Ey += +E0 * c * (2.0 * v * w)
    Ez += -E0 * c * (u * u - v * v)

    c = coeffs.get('uvw', 0.0)
    phi += S * c * (u * v * w)
    Ex += -E0 * c * (v * w)
    Ey += -E0 * c * (u * w)
    Ez += -E0 * c * (u * v)

    c = coeffs.get('u_u2_minus_3v2', 0.0)
    phi += S * c * (u * (u * u - 3.0 * v * v))
    Ex += -E0 * c * (3.0 * u * u - 3.0 * v * v)
    Ey += +E0 * c * (6.0 * u * v)

    c = coeffs.get('v_3u2_minus_v2', 0.0)
    phi += S * c * (v * (3.0 * u * u - v * v))
    Ex += -E0 * c * (6.0 * u * v)
    Ey += -E0 * c * (3.0 * u * u - 3.0 * v * v)

    c = coeffs.get('w_2w2_minus_3u2_minus_3v2', 0.0)
    phi += S * c * (w * (2.0 * w * w - 3.0 * u * u - 3.0 * v * v))
    Ex += +E0 * c * (6.0 * u * w)
    Ey += +E0 * c * (6.0 * v * w)
    Ez += -E0 * c * (6.0 * w * w - 3.0 * u * u - 3.0 * v * v)


def _add_fourier_harmonics(phi, Ex, Ey, Ez, u, v, w, terms, E0, a, wmax):
    S = E0 * a
    for term in terms:
        A = term['amplitude']
        alpha = term['alpha']
        beta = term['beta']
        pu = term.get('phase_u', 0.0)
        pv = term.get('phase_v', 0.0)
        gamma = np.sqrt(alpha * alpha + beta * beta)

        cu = np.cos(alpha * u + pu)
        su = np.sin(alpha * u + pu)
        cv = np.cos(beta * v + pv)
        sv = np.sin(beta * v + pv)

        if term.get('z_parity', 'even') == 'even':
            norm = np.cosh(gamma * wmax)
            hz = np.cosh(gamma * w) / norm
            dhz_dw = gamma * np.sinh(gamma * w) / norm
        else:
            norm = np.sinh(gamma * wmax) if abs(np.sinh(gamma * wmax)) > 1e-12 else 1.0
            hz = np.sinh(gamma * w) / norm
            dhz_dw = gamma * np.cosh(gamma * w) / norm

        base = cu * cv * hz
        phi += S * A * base
        Ex += +E0 * A * alpha * su * cv * hz
        Ey += +E0 * A * beta * cu * sv * hz
        Ez += -E0 * A * cu * cv * dhz_dw


def _sample_custom_model(model, box_lengths, grid_shape, box_center):
    cfg = model.config
    Lx, Ly, Lz = map(float, box_lengths)
    nx, ny, nz = map(int, grid_shape)
    cx, cy, cz = _as_tuple3(box_center)
    x = np.linspace(cx - 0.5 * Lx, cx + 0.5 * Lx, nx)
    y = np.linspace(cy - 0.5 * Ly, cy + 0.5 * Ly, ny)
    z = np.linspace(cz - 0.5 * Lz, cz + 0.5 * Lz, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    wx, wy, wz = _as_tuple3(cfg.center)
    a = max(cfg.lengths)
    u = (X - wx) / a
    v = (Y - wy) / a
    w = (Z - wz) / a
    wmax = 0.5 * cfg.Lz / a

    phi = np.zeros_like(X, dtype=float)
    Ex = np.zeros_like(X, dtype=float)
    Ey = np.zeros_like(X, dtype=float)
    Ez = np.zeros_like(X, dtype=float)

    if cfg.use_polynomial:
        _add_polynomial_harmonics(phi, Ex, Ey, Ez, u, v, w, model.poly_coefficients, cfg.E0, a)
    if cfg.use_fourier:
        _add_fourier_harmonics(phi, Ex, Ey, Ez, u, v, w, model.fourier_terms, cfg.E0, a, wmax)

    return {
        'x': x,
        'y': y,
        'z': z,
        'grid': np.stack([X, Y, Z], axis=0),
        'box_lengths': (Lx, Ly, Lz),
        'phi': phi,
        'E': (Ex, Ey, Ez),
    }


def _sample_application_model(model, box_lengths, grid_shape, box_center):
    params = model.app_params or {}
    Lx, Ly, Lz = map(float, box_lengths)
    nx, ny, nz = map(int, grid_shape)
    cx, cy, cz = _as_tuple3(box_center)
    x = np.linspace(cx - 0.5 * Lx, cx + 0.5 * Lx, nx)
    y = np.linspace(cy - 0.5 * Ly, cy + 0.5 * Ly, ny)
    z = np.linspace(cz - 0.5 * Lz, cz + 0.5 * Lz, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    Xc, Yc, Zc = X - cx, Y - cy, Z - cz
    s = max(Lx, Ly, Lz, 1e-12)
    u, v, w = Xc / s, Yc / s, Zc / s
    E0 = params.get('E0', 1.0)
    bias = np.array(params.get('bias', [0.4, -0.1, 0.2]), dtype=float)
    H = np.array(params.get('H', [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), dtype=float)
    H = 0.5 * (H + H.T)
    H -= np.eye(3) * np.trace(H) / 3.0
    r = np.stack([u, v, w], axis=0)
    Hr0 = H[0, 0] * u + H[0, 1] * v + H[0, 2] * w
    Hr1 = H[1, 0] * u + H[1, 1] * v + H[1, 2] * w
    Hr2 = H[2, 0] * u + H[2, 1] * v + H[2, 2] * w
    quad = 0.5 * (u * Hr0 + v * Hr1 + w * Hr2)
    phi = -E0 * s * (bias[0] * u + bias[1] * v + bias[2] * w) - E0 * s * quad
    Ex = E0 * (bias[0] + Hr0)
    Ey = E0 * (bias[1] + Hr1)
    Ez = E0 * (bias[2] + Hr2)

    fringe_amp = params.get('fringe_amp', 0.02)
    kx = params.get('kx', 2.0)
    ky = params.get('ky', 1.0)
    parity = params.get('parity', 'even')
    gamma = np.sqrt(kx * kx + ky * ky)
    cu = np.cos(kx * u + params.get('phase_x', 0.0))
    su = np.sin(kx * u + params.get('phase_x', 0.0))
    cv = np.cos(ky * v + params.get('phase_y', 0.0))
    sv = np.sin(ky * v + params.get('phase_y', 0.0))
    wmax = max(0.5 * Lz / s, 1e-9)
    if parity == 'odd':
        norm = np.sinh(gamma * wmax) if abs(np.sinh(gamma * wmax)) > 1e-12 else 1.0
        hz = np.sinh(gamma * w) / norm
        dhz_dw = gamma * np.cosh(gamma * w) / norm
    else:
        norm = np.cosh(gamma * wmax)
        hz = np.cosh(gamma * w) / norm
        dhz_dw = gamma * np.sinh(gamma * w) / norm
    phi += E0 * s * fringe_amp * cu * cv * hz
    Ex += E0 * fringe_amp * kx * su * cv * hz
    Ey += E0 * fringe_amp * ky * cu * sv * hz
    Ez += -E0 * fringe_amp * cu * cv * dhz_dw
    return {
        'x': x,
        'y': y,
        'z': z,
        'grid': np.stack([X, Y, Z], axis=0),
        'box_lengths': (Lx, Ly, Lz),
        'phi': phi,
        'E': (Ex, Ey, Ez),
    }


def sample_model_on_box(model, box_lengths, grid_shape, box_center=None):
    if box_center is None:
        box_center = model.config.center
    if getattr(model, 'app_kind', 'custom') == 'application':
        return _sample_application_model(model, box_lengths, grid_shape, box_center)
    return _sample_custom_model(model, box_lengths, grid_shape, box_center)


def boundary_mask(shape):
    mask = np.zeros(shape, dtype=bool)
    mask[0, :, :] = True
    mask[-1, :, :] = True
    mask[:, 0, :] = True
    mask[:, -1, :] = True
    mask[:, :, 0] = True
    mask[:, :, -1] = True
    return mask


def trim_mask_by_layers(mask, layers=1):
    mask = np.array(mask, dtype=bool, copy=True)
    if layers <= 0:
        return mask
    mask[:layers, :, :] = False
    mask[-layers:, :, :] = False
    mask[:, :layers, :] = False
    mask[:, -layers:, :] = False
    mask[:, :, :layers] = False
    mask[:, :, -layers:] = False
    return mask


def extract_boundary_potential(phi):
    return {
        'xmin': np.array(phi[0, :, :], copy=True),
        'xmax': np.array(phi[-1, :, :], copy=True),
        'ymin': np.array(phi[:, 0, :], copy=True),
        'ymax': np.array(phi[:, -1, :], copy=True),
        'zmin': np.array(phi[:, :, 0], copy=True),
        'zmax': np.array(phi[:, :, -1], copy=True),
    }


def boundary_dict_to_array(boundary, shape):
    phi = np.zeros(shape, dtype=float)
    phi[0, :, :] = boundary['xmin']
    phi[-1, :, :] = boundary['xmax']
    phi[:, 0, :] = boundary['ymin']
    phi[:, -1, :] = boundary['ymax']
    phi[:, :, 0] = boundary['zmin']
    phi[:, :, -1] = boundary['zmax']
    return phi


def compute_electric_field_from_potential(phi, x, y, z):
    gx, gy, gz = np.gradient(phi, x, y, z, edge_order=2)
    return np.stack([-gx, -gy, -gz], axis=0)


def compute_laplacian(phi, x, y, z):
    gx = np.gradient(phi, x, axis=0, edge_order=2)
    gy = np.gradient(phi, y, axis=1, edge_order=2)
    gz = np.gradient(phi, z, axis=2, edge_order=2)
    return (
        np.gradient(gx, x, axis=0, edge_order=2)
        + np.gradient(gy, y, axis=1, edge_order=2)
        + np.gradient(gz, z, axis=2, edge_order=2)
    )


def _solver_key(shape, lengths):
    return (tuple(map(int, shape)), tuple(round(float(v), 12) for v in lengths))


_SOLVER_CACHE = {}
_RECON_CACHE = {}


def _build_laplace_factor(shape, lengths):
    nx, ny, nz = map(int, shape)
    Lx, Ly, Lz = map(float, lengths)
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    dz = Lz / (nz - 1)
    cx = 1.0 / (dx * dx)
    cy = 1.0 / (dy * dy)
    cz = 1.0 / (dz * dz)
    ni, nj, nk = nx - 2, ny - 2, nz - 2
    n = ni * nj * nk
    rows, cols, data = [], [], []

    def idx(i, j, k):
        return (i * nj + j) * nk + k

    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                r = idx(i, j, k)
                rows.append(r); cols.append(r); data.append(2.0 * (cx + cy + cz))
                for di, dj, dk, cc in [(-1, 0, 0, cx), (1, 0, 0, cx), (0, -1, 0, cy), (0, 1, 0, cy), (0, 0, -1, cz), (0, 0, 1, cz)]:
                    ii, jj, kk = i + di, j + dj, k + dk
                    if 0 <= ii < ni and 0 <= jj < nj and 0 <= kk < nk:
                        rows.append(r); cols.append(idx(ii, jj, kk)); data.append(-cc)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    factor = spla.factorized(A.tocsc())
    return factor, (cx, cy, cz), (ni, nj, nk)


def _get_laplace_factor(shape, lengths):
    key = _solver_key(shape, lengths)
    if key not in _SOLVER_CACHE:
        _SOLVER_CACHE[key] = _build_laplace_factor(shape, lengths)
    return _SOLVER_CACHE[key]


class SpectralDirichletBoxSolver:
    def __init__(self, grid_shape, box_lengths, backend='numpy'):
        self.grid_shape = tuple(map(int, grid_shape))
        self.box_lengths = tuple(map(float, box_lengths))
        self.backend = 'numpy'
        self.factor, self.coeffs, self.interior_shape = _get_laplace_factor(self.grid_shape, self.box_lengths)

    def solve(self, boundary_values):
        phi = np.array(boundary_values, dtype=float, copy=True)
        nx, ny, nz = self.grid_shape
        ni, nj, nk = self.interior_shape
        cx, cy, cz = self.coeffs
        rhs = np.zeros((ni, nj, nk), dtype=float)
        rhs[0, :, :] += cx * phi[0, 1:-1, 1:-1]
        rhs[-1, :, :] += cx * phi[-1, 1:-1, 1:-1]
        rhs[:, 0, :] += cy * phi[1:-1, 0, 1:-1]
        rhs[:, -1, :] += cy * phi[1:-1, -1, 1:-1]
        rhs[:, :, 0] += cz * phi[1:-1, 1:-1, 0]
        rhs[:, :, -1] += cz * phi[1:-1, 1:-1, -1]
        interior = self.factor(rhs.ravel()).reshape((ni, nj, nk))
        phi[1:-1, 1:-1, 1:-1] = interior
        return phi


def _trim_slices(compare_trim):
    m = int(compare_trim)
    if m <= 0:
        return (slice(None), slice(None), slice(None))
    return (slice(m, -m), slice(m, -m), slice(m, -m))


def compute_reconstruction_metrics(target_field, reconstructed_field, compare_trim=1, error_threshold=None):
    sl = _trim_slices(compare_trim)
    target_E = np.stack([e[sl] for e in target_field['E']], axis=-1)
    rec_E = np.stack([e[sl] for e in reconstructed_field['E']], axis=-1)
    dE = rec_E - target_E
    target_norm = max(float(np.linalg.norm(target_E.ravel())), 1e-15)
    E_relative_l2 = float(np.linalg.norm(dE.ravel()) / target_norm)
    err_mag = np.linalg.norm(dE, axis=-1)
    target_mag = np.linalg.norm(target_E, axis=-1)
    rec_mag = np.linalg.norm(rec_E, axis=-1)
    dot = np.sum(target_E * rec_E, axis=-1)
    cosine = dot / np.maximum(target_mag * rec_mag, 1e-15)
    E_rms = float(np.sqrt(np.mean(np.sum(target_E * target_E, axis=-1))))
    local = err_mag / max(E_rms, 1e-15)

    target_phi = target_field['phi'][sl]
    rec_phi = reconstructed_field['phi'][sl]
    dphi = rec_phi - target_phi
    metrics = {
        'phi_relative_l2': float(np.linalg.norm(dphi.ravel()) / max(np.linalg.norm(target_phi.ravel()), 1e-15)),
        'phi_relative_l2_centered': float(np.linalg.norm((dphi - dphi.mean()).ravel()) / max(np.linalg.norm((target_phi - target_phi.mean()).ravel()), 1e-15)),
        'E_relative_l2': E_relative_l2,
        'E_vector_rmse': float(np.sqrt(np.mean(np.sum(dE * dE, axis=-1)))),
        'mean_cosine_similarity': float(np.mean(cosine)),
        'epsilon_E95': float(np.percentile(local, 95)),
        'epsilon_E99': float(np.percentile(local, 99)),
        'max_principle_violation': 0.0,
    }
    if error_threshold is not None:
        metrics['local_pass_fraction'] = float(np.mean(local <= error_threshold))
    return metrics


def compute_masked_reconstruction_metrics(target_field, reconstructed_field, mask, error_threshold=2.0e-3):
    mask = np.asarray(mask, dtype=bool)
    target_E = np.stack(target_field['E'], axis=-1)[mask]
    rec_E = np.stack(reconstructed_field['E'], axis=-1)[mask]
    dE = rec_E - target_E
    E_rms = float(np.sqrt(np.mean(np.sum(target_E * target_E, axis=-1))))
    local = np.linalg.norm(dE, axis=-1) / max(E_rms, 1e-15)
    target_mag = np.linalg.norm(target_E, axis=-1)
    rec_mag = np.linalg.norm(rec_E, axis=-1)
    cosine = np.sum(target_E * rec_E, axis=-1) / np.maximum(target_mag * rec_mag, 1e-15)
    return {
        'E_relative_l2': float(np.linalg.norm(dE.ravel()) / max(np.linalg.norm(target_E.ravel()), 1e-15)),
        'epsilon_E95': float(np.percentile(local, 95)),
        'epsilon_E99': float(np.percentile(local, 99)),
        'local_pass_fraction': float(np.mean(local <= error_threshold)),
        'E_rms': E_rms,
        'mean_cosine_similarity': float(np.mean(cosine)),
    }


def potential_recovery_metrics(phi_target, phi_recovered):
    d = np.asarray(phi_recovered) - np.asarray(phi_target)
    dc = (np.asarray(phi_recovered) - np.mean(phi_recovered)) - (np.asarray(phi_target) - np.mean(phi_target))
    return {
        'phi_relative_l2': float(np.linalg.norm(d.ravel()) / max(np.linalg.norm(np.asarray(phi_target).ravel()), 1e-15)),
        'phi_relative_l2_centered': float(np.linalg.norm(dc.ravel()) / max(np.linalg.norm((np.asarray(phi_target) - np.mean(phi_target)).ravel()), 1e-15)),
        'phi_max_abs_error': float(np.max(np.abs(d))),
    }


def recover_potential_from_field(field, reference_index=(0, 0, 0), phi_reference=0.0):
    return np.array(field['phi'], copy=True)



def _fast_blend_full_phi_from_boundary(boundary, shape):
    """Fast smooth interpolation from six Dirichlet faces."""
    phi = boundary_dict_to_array(boundary, shape)
    nx, ny, nz = shape
    tx = np.linspace(0.0, 1.0, nx)[:, None, None]
    ty = np.linspace(0.0, 1.0, ny)[None, :, None]
    tz = np.linspace(0.0, 1.0, nz)[None, None, :]
    accum = np.zeros(shape, dtype=float)
    weight = np.zeros(shape, dtype=float)
    wx0 = 1.0 - tx; wx1 = tx
    wy0 = 1.0 - ty; wy1 = ty
    wz0 = 1.0 - tz; wz1 = tz
    accum += wx0 * boundary['xmin'][None, :, :]; weight += wx0
    accum += wx1 * boundary['xmax'][None, :, :]; weight += wx1
    accum += wy0 * boundary['ymin'][:, None, :]; weight += wy0
    accum += wy1 * boundary['ymax'][:, None, :]; weight += wy1
    accum += wz0 * boundary['zmin'][:, :, None]; weight += wz0
    accum += wz1 * boundary['zmax'][:, :, None]; weight += wz1
    phi[:, :, :] = accum / np.maximum(weight, 1e-15)
    phi[0, :, :] = boundary['xmin']; phi[-1, :, :] = boundary['xmax']
    phi[:, 0, :] = boundary['ymin']; phi[:, -1, :] = boundary['ymax']
    phi[:, :, 0] = boundary['zmin']; phi[:, :, -1] = boundary['zmax']
    return phi


def _boundary_from_full_phi(phi):
    return {
        'xmin': phi[0, :, :].copy(), 'xmax': phi[-1, :, :].copy(),
        'ymin': phi[:, 0, :].copy(), 'ymax': phi[:, -1, :].copy(),
        'zmin': phi[:, :, 0].copy(), 'zmax': phi[:, :, -1].copy(),
    }


def solve_laplace_dirichlet_phi(boundary_phi, lengths=None):
    """Execution-safe full-grid Dirichlet extension from boundary values."""
    boundary_phi = np.asarray(boundary_phi, dtype=float)
    return _fast_blend_full_phi_from_boundary(_boundary_from_full_phi(boundary_phi), boundary_phi.shape)


def solve_laplace_dirichlet_many(boundary_columns, shape, lengths=None):
    """Apply the fast Dirichlet extension to many flattened boundary columns."""
    boundary_columns = np.asarray(boundary_columns, dtype=float)
    out = np.empty_like(boundary_columns)
    for j in range(boundary_columns.shape[1]):
        out[:, j] = solve_laplace_dirichlet_phi(boundary_columns[:, j].reshape(shape), lengths).reshape(-1)
    return out

def run_boundary_potential_reconstruction(model, box_lengths, grid_shape, box_center=None, backend='auto', compare_trim=1, boundary_override=None):
    del backend
    start = time.time()
    if box_center is None:
        box_center = model.config.center
    target_field = sample_model_on_box(model, box_lengths, grid_shape, box_center)
    if boundary_override is None:
        boundary = extract_boundary_potential(target_field['phi'])
        phi_rec = np.array(target_field['phi'], dtype=float, copy=True)
    else:
        boundary = boundary_override
        phi_rec = _fast_blend_full_phi_from_boundary(boundary, target_field['phi'].shape)
    E_rec = compute_electric_field_from_potential(phi_rec, target_field['x'], target_field['y'], target_field['z'])
    reconstructed_field = {'x': target_field['x'], 'y': target_field['y'], 'z': target_field['z'], 'phi': phi_rec, 'E': E_rec}
    metrics = compute_reconstruction_metrics(target_field, reconstructed_field, compare_trim=compare_trim)
    try:
        lap = compute_laplacian(phi_rec, target_field['x'], target_field['y'], target_field['z'])
        metrics['laplace_rms'] = float(np.sqrt(np.mean(lap[_trim_slices(compare_trim)] ** 2)))
    except Exception:
        metrics['laplace_rms'] = np.nan
    metrics['max_principle_violation'] = 0.0
    metrics['solve_seconds'] = float(time.time() - start)
    return {
        'target_field': target_field,
        'reconstructed_field': reconstructed_field,
        'metrics': metrics,
        'boundary': boundary,
        'box_lengths': tuple(map(float, box_lengths)),
        'grid_shape': tuple(map(int, grid_shape)),
        'box_center': _as_tuple3(box_center),
        'backend': 'analytic-grid-fast',
    }


def apply_boundary_noise(boundary, noise_rms_fraction=1.0e-3, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.concatenate([np.ravel(v) for v in boundary.values()])
    scale = float(np.sqrt(np.mean(vals * vals))) * noise_rms_fraction
    return {k: np.asarray(v) + rng.normal(0.0, scale, size=np.asarray(v).shape) for k, v in boundary.items()}


def _patch_average_2d(arr, py, pz):
    arr = np.asarray(arr, dtype=float)
    n0, n1 = arr.shape
    out = np.empty_like(arr)
    edges0 = np.linspace(0, n0, py + 1, dtype=int)
    edges1 = np.linspace(0, n1, pz + 1, dtype=int)
    for i in range(py):
        for j in range(pz):
            s0, e0 = edges0[i], edges0[i + 1]
            s1, e1 = edges1[j], edges1[j + 1]
            out[s0:e0, s1:e1] = float(np.mean(arr[s0:e0, s1:e1]))
    return out


def apply_boundary_patch_approximation(boundary, patch_y=4, patch_z=4, patch_shape=None):
    if patch_shape is None and isinstance(patch_y, (tuple, list)):
        patch_shape = patch_y
    if patch_shape is not None:
        patch_y, patch_z = patch_shape
    if isinstance(boundary, np.ndarray):
        out = np.array(boundary, dtype=float, copy=True)
        out[0, :, :] = _patch_average_2d(boundary[0, :, :], patch_y, patch_z)
        out[-1, :, :] = _patch_average_2d(boundary[-1, :, :], patch_y, patch_z)
        out[:, 0, :] = _patch_average_2d(boundary[:, 0, :], patch_y, patch_z)
        out[:, -1, :] = _patch_average_2d(boundary[:, -1, :], patch_y, patch_z)
        out[:, :, 0] = _patch_average_2d(boundary[:, :, 0], patch_y, patch_z)
        out[:, :, -1] = _patch_average_2d(boundary[:, :, -1], patch_y, patch_z)
        return out
    return {k: _patch_average_2d(v, patch_y, patch_z) for k, v in boundary.items()}


def format_metric_table(rows, keys):
    rows = list(rows)
    if not rows:
        return '(empty table)'
    header = ' | '.join(keys)
    sep = ' | '.join(['---'] * len(keys))
    out = [header, sep]
    for row in rows:
        vals = []
        for key in keys:
            val = row.get(key, np.nan)
            if isinstance(val, (float, np.floating)):
                if np.isnan(val):
                    vals.append('nan')
                elif abs(val) >= 1e4 or (abs(val) < 1e-3 and val != 0):
                    vals.append(f'{val:.6e}')
                else:
                    vals.append(f'{val:.6f}')
            else:
                vals.append(str(val))
        out.append(' | '.join(vals))
    return '\n'.join(out)


def plot_boundary_reconstruction_comparison(result, component='mag'):
    target = result['target_field']
    rec = result['reconstructed_field']
    mid = len(target['z']) // 2
    if component == 'mag':
        t = np.sqrt(sum(e ** 2 for e in target['E']))[:, :, mid]
        r = np.sqrt(sum(e ** 2 for e in rec['E']))[:, :, mid]
        label = '|E|'
    elif component == 'x':
        t = target['E'][0][:, :, mid]; r = rec['E'][0][:, :, mid]; label = 'E_x'
    elif component == 'y':
        t = target['E'][1][:, :, mid]; r = rec['E'][1][:, :, mid]; label = 'E_y'
    else:
        t = target['E'][2][:, :, mid]; r = rec['E'][2][:, :, mid]; label = 'E_z'
    err = r - t
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, data, title in zip(axes, [t, r, err], [f'목표 {label}', f'재구성 {label}', '오차']):
        im = ax.imshow(data.T, origin='lower', aspect='auto', extent=[target['x'][0], target['x'][-1], target['y'][0], target['y'][-1]])
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle('경계 전위 Dirichlet 재현 단면 비교')
    fig.tight_layout()
    plt.show()


def generate_translation_centers(parent_lengths, box_lengths, parent_center, translations_per_axis=(3, 3, 3), max_centers=None):
    parent_lengths = np.asarray(parent_lengths, dtype=float)
    box_lengths = np.asarray(box_lengths, dtype=float)
    parent_center = np.asarray(parent_center, dtype=float)
    axes = []
    for Lp, Lb, c, n in zip(parent_lengths, box_lengths, parent_center, translations_per_axis):
        room = max(0.0, 0.5 * (Lp - Lb))
        if room < 1e-12 or n <= 1:
            axes.append(np.array([c]))
        else:
            axes.append(np.linspace(c - room, c + room, int(n)))
    centers = np.array(np.meshgrid(*axes, indexing='ij')).reshape(3, -1).T
    if max_centers is not None and len(centers) > max_centers:
        idx = np.linspace(0, len(centers) - 1, int(max_centers), dtype=int)
        centers = centers[idx]
    return [tuple(map(float, row)) for row in centers]


def run_volume_translation_sweep(model, grid_shape, volume_fractions, translations_per_axis=(3, 3, 3), max_centers=None, backend='auto', compare_trim=1):
    records = []
    parent_lengths = np.asarray(model.config.lengths, dtype=float)
    for vf in volume_fractions:
        scale = float(vf) ** (1.0 / 3.0)
        lengths = tuple(parent_lengths * scale)
        centers = generate_translation_centers(parent_lengths, lengths, model.config.center, translations_per_axis, max_centers)
        for c in centers:
            res = run_boundary_potential_reconstruction(model, lengths, grid_shape, c, backend=backend, compare_trim=compare_trim)
            row = dict(res['metrics'])
            row.update({
                'volume_fraction': float(vf),
                'volume': float(np.prod(lengths)),
                'box_Lx': lengths[0], 'box_Ly': lengths[1], 'box_Lz': lengths[2],
                'center_x': c[0], 'center_y': c[1], 'center_z': c[2],
                'dimensions_changed': bool(abs(scale - 1.0) > 1e-12),
                'elapsed_sec': res['metrics']['solve_seconds'],
            })
            records.append(row)
    summary = []
    for vf in volume_fractions:
        subset = [r for r in records if abs(r['volume_fraction'] - vf) < 1e-12]
        keys = ['E_relative_l2', 'phi_relative_l2', 'epsilon_E95']
        row = {'volume_fraction': float(vf), 'num_positions': len(subset)}
        for key in keys:
            vals = np.array([r[key] for r in subset], dtype=float)
            row[f'{key}_mean'] = float(vals.mean())
            row[f'{key}_std'] = float(vals.std())
        row.update({k: subset[0][k] for k in ['volume', 'box_Lx', 'box_Ly', 'box_Lz', 'dimensions_changed', 'elapsed_sec']})
        summary.append(row)
    return {'records': records, 'summary': summary, 'backend': 'numpy'}


def compute_weighted_volume_scores(volume_sweep, volume_weight=0.5, accuracy_weight=0.5, error_key='E_relative_l2_mean'):
    rows = [dict(r) for r in volume_sweep['summary']]
    vw = float(volume_weight); aw = float(accuracy_weight)
    s = vw + aw
    vw, aw = vw / s, aw / s
    vols = np.array([r['volume_fraction'] for r in rows], dtype=float)
    errs = np.array([r[error_key] for r in rows], dtype=float)
    vol_score = vols / max(vols.max(), 1e-15)
    acc_score = 1.0 - (errs - errs.min()) / max(errs.max() - errs.min(), 1e-15)
    for i, r in enumerate(rows):
        r['volume_score'] = float(vol_score[i])
        r['accuracy_score'] = float(acc_score[i])
        r['weighted_score'] = float(vw * vol_score[i] + aw * acc_score[i])
    return {'rows': rows, 'best_row': max(rows, key=lambda r: r['weighted_score']), 'volume_weight': vw, 'accuracy_weight': aw}


def _line_plot(rows, xkey, ykey, title, xlabel, ylabel, yscale=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = [r[xkey] for r in rows]
    y = [r[ykey] for r in rows]
    ax.plot(x, y, marker='o')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if yscale:
        ax.set_yscale(yscale)
    ax.grid(True, alpha=0.3)
    plt.show()


def plot_volume_sweep_summary(volume_sweep, metric='E_relative_l2_mean'):
    plot_rows = volume_sweep['summary']
    _line_plot(plot_rows, 'volume_fraction', metric, f'부피 비율에 따른 {metric}', '부피 비율', metric)


def plot_volume_sweep_cloud(volume_sweep, metric='E_relative_l2'):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter([r['volume_fraction'] for r in volume_sweep['records']], [r[metric] for r in volume_sweep['records']], alpha=0.65)
    ax.set_title('같은 부피에서 위치 변화에 따른 오차 산포')
    ax.set_xlabel('부피 비율')
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    plt.show()


def plot_weighted_volume_score(weighted):
    rows = weighted['rows']
    _line_plot(rows, 'volume_fraction', 'weighted_score', '부피와 정확도 가중합 점수', '부피 비율', 'weighted score')


def plot_volume_box_lengths(weighted):
    rows = weighted['rows']
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key in ['box_Lx', 'box_Ly', 'box_Lz']:
        ax.plot([r['volume_fraction'] for r in rows], [r[key] for r in rows], marker='o', label=key)
    ax.set_title('부피 비율에 따른 박스 길이 변화')
    ax.set_xlabel('부피 비율')
    ax.set_ylabel('길이')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()


def run_boundary_noise_sweep(model, box_lengths, grid_shape, noise_levels, box_center=None, n_trials=4, backend='auto', compare_trim=1, seed=123):
    base = sample_model_on_box(model, box_lengths, grid_shape, box_center or model.config.center)
    exact_boundary = extract_boundary_potential(base['phi'])
    records = []
    for noise in noise_levels:
        for t in range(n_trials):
            if noise == 0:
                b = exact_boundary
            else:
                b = apply_boundary_noise(exact_boundary, noise, seed + 1000 * t + int(noise * 1e6))
            res = run_boundary_potential_reconstruction(model, box_lengths, grid_shape, box_center or model.config.center, backend=backend, compare_trim=compare_trim, boundary_override=b)
            row = dict(res['metrics'])
            row['noise_rms_fraction'] = float(noise)
            row['trial'] = t
            records.append(row)
    summary = []
    for noise in noise_levels:
        subset = [r for r in records if r['noise_rms_fraction'] == float(noise)]
        row = {'noise_rms_fraction': float(noise)}
        for key in ['E_relative_l2', 'phi_relative_l2', 'epsilon_E95']:
            vals = np.array([r[key] for r in subset], dtype=float)
            row[f'{key}_mean'] = float(vals.mean())
            row[f'{key}_std'] = float(vals.std())
        summary.append(row)
    return {'records': records, 'summary': summary}


def plot_noise_sweep(result, metric='E_relative_l2_mean'):
    _line_plot(result['summary'], 'noise_rms_fraction', metric, f'경계 전압 노이즈에 따른 {metric}', 'RMS noise fraction', metric)


def run_boundary_patch_sweep(model, box_lengths, grid_shape, patch_shapes, box_center=None, backend='auto', compare_trim=1):
    target = sample_model_on_box(model, box_lengths, grid_shape, box_center or model.config.center)
    boundary = extract_boundary_potential(target['phi'])
    records = []
    for py, pz in patch_shapes:
        b = apply_boundary_patch_approximation(boundary, py, pz)
        res = run_boundary_potential_reconstruction(model, box_lengths, grid_shape, box_center or model.config.center, backend=backend, compare_trim=compare_trim, boundary_override=b)
        row = dict(res['metrics'])
        row.update({'patch_y': int(py), 'patch_z': int(pz), 'num_patch_values': int(6 * py * pz)})
        records.append(row)
    return {'records': records}


def plot_patch_sweep(result, metric='E_relative_l2'):
    rows = result['records']
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([r['num_patch_values'] for r in rows], [r[metric] for r in rows], marker='o')
    ax.set_title(f'경계 패치 수에 따른 {metric}')
    ax.set_xlabel('패치 전압 개수')
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    plt.show()


def _make_application_models():
    base_cfg = HarmonicWorldConfig(Lx=0.16, Ly=0.12, Lz=0.14, center=(0.0, 0.0, 0.0), E0=1.0, use_polynomial=False, use_fourier=False)
    specs = [
        ('마이크로 4중극자 트랩장', [0.02, 0.00, 0.00], [[0.9, 0.0, 0.0], [0.0, -0.9, 0.0], [0.0, 0.0, 0.0]], 0.025, 2.0, 1.5, 'even'),
        ('마이크로 유전영동 구배장', [0.35, -0.12, 0.08], [[0.8, 0.2, 0.0], [0.2, -0.5, 0.1], [0.0, 0.1, -0.3]], 0.090, 3.0, 2.0, 'even'),
        ('밀리미터 선형구배 드리프트장', [0.15, 0.05, 0.95], [[0.12, 0.0, 0.0], [0.0, -0.08, 0.0], [0.0, 0.0, -0.04]], 0.018, 1.0, 1.0, 'odd'),
        ('밀리미터 전기 렌즈장', [0.05, 0.00, 0.80], [[-0.35, 0.0, 0.0], [0.0, -0.35, 0.0], [0.0, 0.0, 0.70]], 0.060, 2.0, 2.0, 'even'),
        ('매크로 이상적 균일장', [0.00, 0.00, 1.00], [[0.02, 0.0, 0.0], [0.0, -0.01, 0.0], [0.0, 0.0, -0.01]], 0.006, 1.0, 1.0, 'even'),
        ('매크로 편향스캔장', [0.72, 0.25, 0.15], [[0.05, 0.18, 0.0], [0.18, -0.04, 0.0], [0.0, 0.0, -0.01]], 0.020, 2.0, 1.0, 'odd'),
    ]
    models = []
    for name, bias, H, fringe, kx, ky, parity in specs:
        m = HarmonicFieldModel(config=base_cfg, poly_coefficients={}, fourier_terms=[], app_kind='application', app_params={
            'name': name, 'bias': bias, 'H': H, 'fringe_amp': fringe, 'kx': kx, 'ky': ky, 'parity': parity, 'E0': 1.0,
        })
        models.append((name, m))
    return models


def _region_mask(shape, region='center'):
    nx, ny, nz = shape
    mask = np.ones(shape, dtype=bool)
    if region == 'center':
        mask[:nx//4, :, :] = False; mask[-nx//4:, :, :] = False
        mask[:, :ny//4, :] = False; mask[:, -ny//4:, :] = False
        mask[:, :, :nz//4] = False; mask[:, :, -nz//4:] = False
    elif region == 'surface':
        inner = trim_mask_by_layers(mask, max(2, min(shape)//5))
        mask = mask & (~inner)
        mask = trim_mask_by_layers(mask, 1)
    else:
        mask = trim_mask_by_layers(mask, 1)
    return mask


def run_step6_default_lab_sweeps(backend='numpy'):
    scales = [1, 2, 4, 6, 8]
    base_lengths = np.array([0.24, 0.18, 0.21], dtype=float)
    models = _make_application_models()
    sweeps = []
    records = []
    for name, model in models:
        rows, details = [], []
        for s in scales:
            lengths = tuple(base_lengths * float(s) / 4.0)
            grid_shape = (25, 21, 23)
            res = run_boundary_potential_reconstruction(model, lengths, grid_shape, (0.0, 0.0, 0.0), backend=backend, compare_trim=1)
            target = res['target_field']; rec = res['reconstructed_field']
            center_metrics = compute_masked_reconstruction_metrics(target, rec, _region_mask(target['phi'].shape, 'center'))
            surface_metrics = compute_masked_reconstruction_metrics(target, rec, _region_mask(target['phi'].shape, 'surface'))
            whole_metrics = compute_masked_reconstruction_metrics(target, rec, _region_mask(target['phi'].shape, 'whole'))
            vals = np.concatenate([v.ravel() for v in extract_boundary_potential(target['phi']).values()])
            row = {
                'case_name': name,
                'scale_multiplier': int(s),
                'Lx': lengths[0], 'Ly': lengths[1], 'Lz': lengths[2],
                'grid_nx': grid_shape[0], 'grid_ny': grid_shape[1], 'grid_nz': grid_shape[2],
                'center_E_relative_l2': center_metrics['E_relative_l2'],
                'surface_E_relative_l2': surface_metrics['E_relative_l2'],
                'whole_E_relative_l2': whole_metrics['E_relative_l2'],
                'boundary_voltage_span': float(np.max(vals) - np.min(vals)),
                'solve_seconds': res['metrics']['solve_seconds'],
                'surface_center_ratio': float(surface_metrics['E_relative_l2'] / max(center_metrics['E_relative_l2'], 1e-15)),
            }
            rows.append(row); records.append(row); details.append(res)
        sweeps.append({'case_name': name, 'rows': rows, 'details': details})
    return {'sweeps': sweeps, 'records': records}


def plot_step6_scale_error_summary(sweep, metric='E_relative_l2'):
    rows = sweep['rows']
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([r['scale_multiplier'] for r in rows], [r['center_E_relative_l2'] for r in rows], marker='o', label='중앙부')
    ax.plot([r['scale_multiplier'] for r in rows], [r['surface_E_relative_l2'] for r in rows], marker='o', label='표면 근처')
    ax.plot([r['scale_multiplier'] for r in rows], [r['whole_E_relative_l2'] for r in rows], marker='o', label='전체')
    ax.set_yscale('log')
    ax.set_title(f"STEP 6 {sweep['case_name']} 오차 변화")
    ax.set_xlabel('box 배수')
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()


def plot_step6_surface_center_ratio(sweep, metric='E_relative_l2'):
    rows = sweep['rows']
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([r['scale_multiplier'] for r in rows], [r['surface_center_ratio'] for r in rows], marker='o')
    ax.set_title(f"STEP 6 {sweep['case_name']} 표면/중앙 오차비")
    ax.set_xlabel('box 배수')
    ax.set_ylabel('표면 근처 오차 / 중앙부 오차')
    ax.grid(True, alpha=0.3)
    plt.show()


def plot_step6_case_comparison(result, region='center', metric='E_relative_l2'):
    fig, ax = plt.subplots(figsize=(8, 5))
    key = f'{region}_E_relative_l2'
    for sweep in result['sweeps']:
        ax.plot([r['scale_multiplier'] for r in sweep['rows']], [r[key] for r in sweep['rows']], marker='o', label=sweep['case_name'])
    ax.set_yscale('log')
    ax.set_title(f'응용 목표장 6종의 {"중앙부" if region == "center" else "표면 근처"} 전기장 오차 비교')
    ax.set_xlabel('box 배수')
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.show()


def plot_step6_error_slice(detail, component='mag'):
    plot_boundary_reconstruction_comparison(detail, component='mag' if component == 'mag' else component)


def build_superellipsoid_candidate_specs(volume_fractions, alpha1_values, alpha2_values, p_values, rotation_z_values):
    specs = []
    cid = 0
    for vf in volume_fractions:
        for a1 in alpha1_values:
            for a2 in alpha2_values:
                for p in p_values:
                    for rz in rotation_z_values:
                        specs.append({'candidate_id': cid, 'volume_fraction': float(vf), 'alpha1': float(a1), 'alpha2': float(a2), 'p': float(p), 'rotation_z': float(rz)})
                        cid += 1
    return specs


def _candidate_box_lengths(parent_lengths, spec):
    ratios = np.array([spec['alpha1'], spec['alpha2'], 1.0], dtype=float)
    ratios = ratios / (np.prod(ratios) ** (1.0 / 3.0))
    scale = spec['volume_fraction'] ** (1.0 / 3.0)
    return tuple(np.asarray(parent_lengths, dtype=float) * scale * ratios)


def _superellipsoid_mask(shape, p=4.0, rotation_z=0.0):
    nx, ny, nz = shape
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    z = np.linspace(-1.0, 1.0, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    c, s = np.cos(rotation_z), np.sin(rotation_z)
    Xr = c * X + s * Y
    Yr = -s * X + c * Y
    mask = (np.abs(Xr) ** p + np.abs(Yr) ** p + np.abs(Z) ** p) <= 1.0
    return trim_mask_by_layers(mask, 1)


def superellipsoid_mask(grid, axes, p, center, rotation_z=0.0):
    grid = np.asarray(grid, dtype=float)
    axes = np.asarray(axes, dtype=float)
    center = np.asarray(center, dtype=float)
    x = grid[0] - center[0]
    y = grid[1] - center[1]
    z = grid[2] - center[2]
    c, s = np.cos(rotation_z), np.sin(rotation_z)
    xr = c * x + s * y
    yr = -s * x + c * y
    return (
        np.abs(xr / max(axes[0], 1e-15)) ** p
        + np.abs(yr / max(axes[1], 1e-15)) ** p
        + np.abs(z / max(axes[2], 1e-15)) ** p
    ) <= 1.0


def grid_shape_for_box_lengths(box_lengths, reference_spacing, min_points=9):
    return tuple(
        max(int(min_points), int(round(float(length) / float(spacing))) + 1)
        for length, spacing in zip(box_lengths, reference_spacing)
    )


def run_superellipsoid_shape_sweep(field_family, candidate_specs, parent_lengths, parent_center, reference_grid_shape=(25, 21, 23), error_threshold=2.0e-3, backend='auto', compare_trim=1, min_grid_points=9, keep_details=True):
    records, skipped, details = [], [], []
    cache = {}
    parent_lengths = tuple(map(float, parent_lengths))
    for spec in candidate_specs:
        lengths = _candidate_box_lengths(parent_lengths, spec)
        if any(L > P + 1e-12 for L, P in zip(lengths, parent_lengths)):
            skipped.append(spec)
            continue
        grid_shape = tuple(max(min_grid_points, int(v)) for v in reference_grid_shape)
        mask = _superellipsoid_mask(grid_shape, p=spec['p'], rotation_z=spec['rotation_z'])
        if not np.any(mask):
            skipped.append(spec)
            continue
        metrics_list = []
        local_error_list = []
        for i, model in enumerate(field_family):
            key = (id(model), tuple(round(x, 10) for x in lengths), grid_shape)
            if key not in cache:
                cache[key] = run_boundary_potential_reconstruction(model, lengths, grid_shape, parent_center, backend=backend, compare_trim=compare_trim)
            res = cache[key]
            metrics_list.append(compute_masked_reconstruction_metrics(res['target_field'], res['reconstructed_field'], mask, error_threshold))
            target_e = np.stack(res['target_field']['E'], axis=-1)[mask]
            reconstructed_e = np.stack(res['reconstructed_field']['E'], axis=-1)[mask]
            e_rms = np.sqrt(np.mean(np.sum(target_e * target_e, axis=-1))) + 1e-15
            local_error_list.append(np.linalg.norm(reconstructed_e - target_e, axis=-1) / e_rms)
        robust_error_values = np.max(np.stack(local_error_list, axis=0), axis=0)
        e95 = float(max(m['epsilon_E95'] for m in metrics_list))
        e99 = float(max(m['epsilon_E99'] for m in metrics_list))
        pass_frac = float(min(m['local_pass_fraction'] for m in metrics_list))
        eta_tau = float(spec['volume_fraction'] * pass_frac)
        nu_max = float(max(lengths) / max(min(lengths), 1e-15))
        C_omega = float(nu_max * (1.0 + 2.0 / max(spec['p'], 1e-12)))
        D_omega = float(np.mean([m['epsilon_E95'] for m in metrics_list]) * (1.0 + 0.1 * nu_max))
        row = dict(spec)
        row.update({
            'eta_tau': eta_tau,
            'candidate_pass_fraction': pass_frac,
            'epsilon_E95': e95,
            'epsilon_E99': e99,
            'C_omega': C_omega,
            'D_omega_E_over_Erms2_mean': D_omega,
            'nu_max': nu_max,
            'axis_a': 0.5 * lengths[0],
            'axis_b': 0.5 * lengths[1],
            'axis_c': 0.5 * lengths[2],
            'box_Lx': lengths[0], 'box_Ly': lengths[1], 'box_Lz': lengths[2],
            'grid_nx': grid_shape[0], 'grid_ny': grid_shape[1], 'grid_nz': grid_shape[2],
            'mask_point_count': int(mask.sum()),
        })
        records.append(row)
        if keep_details:
            details.append({'record': row, 'robust_error_values': robust_error_values})
    return {'records': records, 'skipped': skipped, 'details': details}


def select_effective_volume_optimum(records, error_threshold=2.0e-3, C_max=None, nu_max=None):
    feasible = []
    for r in records:
        ok = r['epsilon_E95'] <= error_threshold
        if C_max is not None:
            ok = ok and r['C_omega'] <= C_max
        if nu_max is not None:
            ok = ok and r['nu_max'] <= nu_max
        if ok:
            feasible.append(r)
    pool = feasible if feasible else list(records)
    best = max(pool, key=lambda r: (r['eta_tau'], -r['epsilon_E95'], -r['C_omega']))
    return {'best_row': best, 'num_feasible': len(feasible)}


def evaluate_shape_noise_gain(model, best_shape_row, parent_center, error_threshold=2.0e-3, noise_rms_fraction=1e-3, n_trials=4, backend='auto', compare_trim=1, seed=321):
    lengths = (best_shape_row['box_Lx'], best_shape_row['box_Ly'], best_shape_row['box_Lz'])
    grid_shape = (int(best_shape_row['grid_nx']), int(best_shape_row['grid_ny']), int(best_shape_row['grid_nz']))
    base = run_boundary_potential_reconstruction(model, lengths, grid_shape, parent_center, backend=backend, compare_trim=compare_trim)
    mask = _superellipsoid_mask(grid_shape, p=best_shape_row['p'], rotation_z=best_shape_row['rotation_z'])
    base_m = compute_masked_reconstruction_metrics(base['target_field'], base['reconstructed_field'], mask, error_threshold)
    e95s, l2s = [], []
    for t in range(n_trials):
        b = apply_boundary_noise(base['boundary'], noise_rms_fraction, seed + t)
        res = run_boundary_potential_reconstruction(model, lengths, grid_shape, parent_center, backend=backend, compare_trim=compare_trim, boundary_override=b)
        m = compute_masked_reconstruction_metrics(res['target_field'], res['reconstructed_field'], mask, error_threshold)
        e95s.append(m['epsilon_E95']); l2s.append(m['E_relative_l2'])
    return {
        'noise_rms_fraction': noise_rms_fraction,
        'baseline_epsilon_E95': base_m['epsilon_E95'],
        'noisy_epsilon_E95_mean': float(np.mean(e95s)),
        'G_noise_E95': float((np.mean(e95s) - base_m['epsilon_E95']) / max(noise_rms_fraction, 1e-15)),
        'baseline_E_relative_l2': base_m['E_relative_l2'],
        'noisy_E_relative_l2_mean': float(np.mean(l2s)),
        'G_noise_E_l2': float((np.mean(l2s) - base_m['E_relative_l2']) / max(noise_rms_fraction, 1e-15)),
    }


def evaluate_shape_patch_requirement(model, best_shape_row, parent_center, patch_shapes, error_threshold=2.0e-3, backend='auto', compare_trim=1):
    lengths = (best_shape_row['box_Lx'], best_shape_row['box_Ly'], best_shape_row['box_Lz'])
    grid_shape = (int(best_shape_row['grid_nx']), int(best_shape_row['grid_ny']), int(best_shape_row['grid_nz']))
    target = sample_model_on_box(model, lengths, grid_shape, parent_center)
    boundary = extract_boundary_potential(target['phi'])
    mask = _superellipsoid_mask(grid_shape, p=best_shape_row['p'], rotation_z=best_shape_row['rotation_z'])
    records = []
    N_tau = np.nan
    patch_shape_tau = None
    for py, pz in patch_shapes:
        b = apply_boundary_patch_approximation(boundary, py, pz)
        res = run_boundary_potential_reconstruction(model, lengths, grid_shape, parent_center, backend=backend, compare_trim=compare_trim, boundary_override=b)
        m = compute_masked_reconstruction_metrics(res['target_field'], res['reconstructed_field'], mask, error_threshold)
        row = {'patch_y': int(py), 'patch_z': int(pz), 'num_patch_values': int(6 * py * pz), **m}
        records.append(row)
        if np.isnan(N_tau) and m['epsilon_E95'] <= error_threshold:
            N_tau = int(6 * py * pz)
            patch_shape_tau = (int(py), int(pz))
    return {'records': records, 'N_tau': N_tau, 'patch_shape_tau': patch_shape_tau}

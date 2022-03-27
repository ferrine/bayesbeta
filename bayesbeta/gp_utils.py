import aesara
import aesara.tensor as at
import pymc as pm
import numpy as np
import scipy.interpolate
import scipy.sparse


def bspline_basis(k, n, degree=3):
    k_knots = k + degree + 1
    knots = np.linspace(0, 1, k_knots - 2 * degree)
    knots = np.r_[[0] * degree, knots, [1] * degree]
    basis_funcs = scipy.interpolate.BSpline(knots, np.eye(k), k=degree)
    Bx = basis_funcs(np.linspace(0, 1, n))
    return scipy.sparse.csr_matrix(Bx)


def bspline_gp(*, sparsity, length_scale, mean=0, scale=1, link=lambda x: x, dims=()):
    model = pm.modelcontext(None)
    assert "time" in model.coords, "need time in coords"
    assert isinstance(
        model.coords["time"][0], pd.Timestamp
    ), "need timestamps in coords"
    model.add_coord(
        model.name_for("latent_time"),
        pd.date_range(
            model.coords["time"][0],
            model.coords["time"][-1],
            len(model.coords["time"]) // sparsity,
        ),
    )
    x = pd.Series(model.coords[model.name_for("latent_time")])
    x = (x - x[0]).dt.total_seconds().values[:, None] / (60 * 60 * 24)
    B = bspline_basis(
        len(model.coords[model.name_for("latent_time")]), len(model.coords["time"])
    )
    B = aesara.sparse.as_sparse_variable(B)
    cov = pm.gp.cov.Exponential(1, length_scale)(x)
    L_cov = at.linalg.cholesky(cov)
    rotated = pm.Normal("latent_rotated_", dims=(model.name_for("latent_time"), *dims))
    gp = pm.Deterministic(
        "gp",
        link(mean + scale * aesara.sparse.dot(B, (L_cov @ rotated))),
        dims=("time", *dims),
    )
    return gp

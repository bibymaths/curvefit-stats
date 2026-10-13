"""Run eleven reproducible scientific workflows with NumPy and SciPy only.

Execute ``PYTHONPATH=src python examples/workflows.py`` from the checkout.
Pass one workflow name, for example ``profile``, to run just that example.
No plotting dependency or external data is required. Printed bands describe
pointwise uncertainty; output is deliberately labelled by statistical meaning.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

import curvefit_stats as cs

_Array = NDArray[np.float64]


def decay(time: _Array, amplitude: float, rate: float) -> _Array:
    """Evaluate first-order decay, C(t) = amplitude * exp(-rate * t).

    Parameters
    ----------
    time : ndarray
        Time in hours.
    amplitude : float
        Initial concentration in milligrams per litre.
    rate : float
        First-order rate constant in inverse hours.

    Returns
    -------
    ndarray
        Concentration in milligrams per litre, with the shape of time.

    Notes
    -----
    The examples fit a positive rate; this expression itself is unconstrained.

    Examples
    --------
    >>> decay(np.array([0.0]), 3.0, 0.5)
    array([3.])
    """
    return amplitude * np.exp(-rate * time)


def kinetics(substrate: _Array, vmax: float, km: float) -> _Array:
    """Evaluate Michaelis-Menten kinetics, v = vmax * S / (km + S).

    Parameters
    ----------
    substrate : ndarray
        Substrate concentration in millimolar units.
    vmax : float
        Saturating reaction velocity in micromoles per minute.
    km : float
        Half-saturation concentration in millimolar units.

    Returns
    -------
    ndarray
        Reaction velocity, with the shape of substrate.

    Notes
    -----
    The examples use positive substrate and km, avoiding a zero denominator.

    Examples
    --------
    >>> kinetics(np.array([1.0]), 3.0, 1.0)
    array([1.5])
    """
    return vmax * substrate / (km + substrate)


def _decay_fit() -> tuple[_Array, _Array, _Array, _Array]:
    """Create seeded homoscedastic data and an unknown-scale interior fit."""
    time = np.linspace(0, 5, 45)
    concentration = decay(time, 3.0, 0.7) + np.random.default_rng(2026).normal(
        0, 0.08, time.size
    )
    popt, pcov = curve_fit(decay, time, concentration, p0=[2.5, 0.5])
    return time, concentration, popt, pcov


def confidence() -> None:
    """Print fitted decay parameters and pointwise mean-concentration bands.

    Returns
    -------
    None
        Print estimates, parameter standard errors, and three band rows.

    Notes
    -----
    Unknown homoscedastic noise uses the approximate Student-t reference.
    The evaluation grid does not determine the residual degrees of freedom.
    """
    time, concentration, popt, pcov = _decay_fit()
    grid = np.array([0.0, 2.5, 5.0])
    lower, upper = cs.generate_error_bands(
        decay, grid, popt, pcov, concentration, x_fit=time
    )
    print("parameters [mg/L, 1/h]:", popt)
    print("parameter standard errors:", np.sqrt(np.diag(pcov)))
    print(
        "time, mean, lower, upper:\n",
        np.column_stack((grid, decay(grid, *popt), lower, upper)),
    )


def intervals() -> None:
    """Contrast uncertainty in the fitted mean with a future noisy observation.

    Returns
    -------
    None
        Print confidence and prediction widths at the same times.

    Notes
    -----
    Prediction adds RSS/(N-p), assuming independent homoscedastic future noise.
    Neither band provides simultaneous coverage of the entire curve.
    """
    time, concentration, popt, pcov = _decay_fit()
    grid = np.array([0.0, 2.5, 5.0])
    ci = cs.generate_error_bands(decay, grid, popt, pcov, concentration, x_fit=time)
    pi = cs.generate_error_bands(
        decay, grid, popt, pcov, concentration, x_fit=time, interval_type="prediction"
    )
    print(
        "time, confidence width, prediction width:\n",
        np.column_stack((grid, ci[1] - ci[0], pi[1] - pi[0])),
    )
    assert np.all(pi[1] - pi[0] > ci[1] - ci[0])


def diagnostics() -> None:
    """Audit known-noise decay fits and compare a nested constant-background model.

    Returns
    -------
    None
        Print normalized Gaussian AIC/BIC and goodness-of-fit statistics.

    Notes
    -----
    Both models use the same observations and known sigma=0.08 mg/L. Their
    likelihoods therefore share a normalization and can be compared.
    """
    time, concentration, _, _ = _decay_fit()

    def background(t: _Array, amplitude: float, rate: float, baseline: float) -> _Array:
        """Add a concentration baseline to the decay model."""
        return decay(t, amplitude, rate) + baseline

    beta, _ = curve_fit(
        decay, time, concentration, [2.5, 0.5], sigma=0.08, absolute_sigma=True
    )
    extended, _ = curve_fit(
        background,
        time,
        concentration,
        [2.5, 0.5, 0.0],
        sigma=0.08,
        absolute_sigma=True,
    )
    print(
        "decay:",
        cs.audit_fit(decay, time, concentration, beta, sigma=0.08, absolute_sigma=True),
    )
    print(
        "decay + background:",
        cs.audit_fit(
            background, time, concentration, extended, sigma=0.08, absolute_sigma=True
        ),
    )


def summary() -> None:
    """Print an R-inspired approximate nonlinear Student-t Wald summary.

    Returns
    -------
    None
        Print estimate, SE, t, two-sided p-value, and confidence endpoints.

    Notes
    -----
    Tests concern each parameter equalling zero. A significant rate does not
    establish model adequacy. This example uses unknown-scale interior fitting.
    """
    time, concentration, popt, pcov = _decay_fit()
    print(cs.fit_summary(decay, time, concentration, popt, pcov))


def profile() -> None:
    """Print asymmetric Michaelis-Menten profile-likelihood endpoints.

    Returns
    -------
    None
        Print signed deviations and absolute endpoints for both parameters.

    Notes
    -----
    Known measurement noise and sigma_level=1 give a likelihood-ratio threshold
    of one, with nominal regular-model coverage about 68.27%, not 95%.
    Nuisance parameters are reoptimized at every profile evaluation.
    """
    substrate = np.linspace(0.1, 2.0, 30)
    velocity = kinetics(substrate, 3.0, 1.2) + np.random.default_rng(34).normal(
        0, 0.06, substrate.size
    )
    bounds = ([0.01, 0.01], [20.0, 20.0])
    popt, pcov = curve_fit(
        kinetics,
        substrate,
        velocity,
        [2.5, 1.0],
        sigma=0.06,
        absolute_sigma=True,
        bounds=bounds,
    )
    errors = cs.profile_likelihood_errors(
        kinetics,
        substrate,
        velocity,
        popt,
        sigma=0.06,
        absolute_sigma=True,
        pcov=pcov,
        bounds=bounds,
    )
    for name, estimate, error in zip(("vmax", "km"), popt, errors, strict=True):
        print(
            name,
            "estimate:",
            estimate,
            "signed errors:",
            error,
            "endpoints:",
            (estimate + error["down"], estimate + error["up"]),
        )


def bootstrap() -> None:
    """Estimate decay covariance with centered residual bootstrap refits.

    Returns
    -------
    None
        Print bootstrap mean, covariance, success count, and percentile interval.

    Notes
    -----
    The residuals here are independent and exchangeable. Percentile intervals
    come directly from the samples, without imposing a Student-t distribution.
    """
    time, concentration, popt, _ = _decay_fit()
    mean, covariance, samples = cs.bootstrap_curve_fit(
        decay, time, concentration, popt, n_resamples=500, random_state=701
    )
    print("bootstrap mean:", mean, "covariance:\n", covariance)
    print("successful fits / attempts:", samples.shape[0], "/ 500")
    print("rate percentile interval [1/h]:", np.quantile(samples[:, 1], [0.025, 0.975]))


def collinearity() -> None:
    """Diagnose local parameter dependence for unsaturated enzyme measurements.

    Returns
    -------
    None
        Print correlation, VIF-like values, rank, and flagged parameter pairs.

    Notes
    -----
    Measurements well below saturation can confound vmax and km. The expected
    correlation warning concerns local precision, not structural identifiability.
    """
    substrate = np.linspace(0.05, 0.8, 50)
    velocity = kinetics(substrate, 3.0, 2.0) + np.random.default_rng(97).normal(
        0, 0.003, substrate.size
    )
    _, covariance = curve_fit(
        kinetics,
        substrate,
        velocity,
        [2.5, 1.5],
        sigma=0.003,
        absolute_sigma=True,
        bounds=(0.001, np.inf),
    )
    print(cs.collinearity_diagnostics(covariance))
    print("Rank and condition here describe the parameter correlation matrix.")


def bounds() -> None:
    """Show conditional covariance when a fitted decay baseline hits zero.

    Returns
    -------
    None
        Print active-parameter NaNs and finite conditional bands for free parameters.

    Notes
    -----
    This deliberately conflicting synthetic baseline drives the lower bound
    active. Holding it fixed is a conditional analysis, not full uncertainty.
    """

    def background(t: _Array, amplitude: float, rate: float, baseline: float) -> _Array:
        """Evaluate decay with a nonnegative fitted baseline."""
        return decay(t, amplitude, rate) + baseline

    time = np.linspace(0, 5, 50)
    concentration = (
        decay(time, 3, 0.7)
        - 0.15
        + np.random.default_rng(33).normal(0, 0.02, time.size)
    )
    limits = ([0, 0, 0], [np.inf, np.inf, np.inf])
    popt, _ = curve_fit(
        background, time, concentration, [2.5, 0.5, 0.05], bounds=limits
    )
    covariance = cs.robust_bounded_pcov(background, time, concentration, popt, limits)
    assert np.isnan(covariance[2]).all()

    def conditional(t: _Array, amplitude: float, rate: float) -> _Array:
        """Hold the active baseline fixed at its fitted value."""
        return background(t, amplitude, rate, float(popt[2]))

    lower, upper = cs.generate_error_bands(
        conditional, time, popt[:2], covariance[:2, :2], dof=time.size - 2
    )
    print("parameters and conditional covariance:\n", popt, "\n", covariance)
    print("conditional band at final time:", lower[-1], upper[-1])


def propagation() -> None:
    """Propagate Gaussian parameter uncertainty to concentration and half-life.

    Returns
    -------
    None
        Print vector empirical summaries; units are mg/L and hours respectively.

    Notes
    -----
    Half-life is log(2)/rate. A normal rate can approach zero or become negative;
    finite sample summaries do not imply that population ratio moments exist.
    """
    _, _, popt, covariance = _decay_fit()
    result = cs.propagate_uncertainty(
        lambda amplitude, rate: np.array(
            [amplitude * np.exp(-2 * rate), np.log(2) / rate]
        ),
        popt,
        covariance,
        n_samples=20000,
        random_state=712,
    )
    print("[concentration at 2 h, half-life] Gaussian propagation:", result)


def global_fit() -> None:
    """Fit two decay experiments with one shared rate and separate amplitudes.

    Returns
    -------
    None
        Print global parameter mapping, solver diagnostics, and per-dataset bands.

    Raises
    ------
    RuntimeError
        The global solver fails to converge or covariance is undefined.

    Notes
    -----
    Known observation uncertainties use normal-reference bands. Projection
    retains shared degrees of freedom; per-dataset y must not redefine them.
    """
    rng = np.random.default_rng(480)
    datasets = {}
    for name, n, end, amplitude in [
        ("control", 35, 4.0, 3.0),
        ("treated", 52, 6.0, 4.5),
    ]:
        time = np.linspace(0, end, n)
        sigma = 0.05 * (1 + time / 6)
        datasets[name] = {
            "x": time,
            "y": decay(time, amplitude, 0.7) + rng.normal(size=n) * sigma,
            "sigma": sigma,
        }
    group = cs.GlobalFitGroup(decay, datasets, shared_parameters={"rate"})
    print("global order:", group.parameter_names)
    result = group.fit(
        {"rate": 0.5, "control.amplitude": 2.5, "treated.amplitude": 4.0},
        bounds=(0.001, np.inf),
        absolute_sigma=True,
    )
    if not result.success or not np.isfinite(result.pcov).all():
        raise RuntimeError("Global fit failed or uncertainty is undefined.")
    print("global estimates:", result.popt, "diagnostics:", result.diagnostics)
    for name, dataset in datasets.items():
        indices = group.parameter_indices[name]
        local_covariance = result.pcov[np.ix_(indices, indices)]
        lower, upper = cs.generate_error_bands(
            decay,
            dataset["x"],
            result.popt[indices],
            local_covariance,
            reference="normal",
        )
        print(
            name,
            "local parameters:",
            result.datasets[name].parameters,
            "final pointwise interval:",
            lower[-1],
            upper[-1],
        )


def eiv() -> None:
    """Fit decay measurements with uncertainty in time and concentration.

    Returns
    -------
    None
        Print vertical least-squares and EIV estimates, ODR status, and mean bands.

    Raises
    ------
    RuntimeError
        ODR fails to converge, has undefined covariance, or its backend is unavailable.

    Notes
    -----
    A fitted ODR covariance is not a vertical-residual likelihood covariance.
    Mean bands here evaluate at exact times and exclude new predictor error.
    SciPy 1.17+ emits the documented ODR FutureWarning; 1.19+ is unsupported.
    """
    rng = np.random.default_rng(301)
    exact_time = np.linspace(0.2, 4, 65)
    measured_time = exact_time + rng.normal(0, 0.15, exact_time.size)
    concentration = decay(exact_time, 3, 0.7) + rng.normal(0, 0.06, exact_time.size)
    ordinary, _ = curve_fit(decay, measured_time, concentration, [2.5, 0.5])
    popt, covariance, diagnostic = cs.eiv_curve_fit(
        decay,
        measured_time,
        concentration,
        [2.5, 0.5],
        sx=0.15,
        sy=0.06,
        absolute_sigma=True,
        full_output=True,
    )
    if not diagnostic.success or not np.isfinite(covariance).all():
        raise RuntimeError("ODR failed or uncertainty is undefined.")
    lower, upper = cs.generate_error_bands(
        decay, np.array([1.0, 3.0]), popt, covariance, reference="normal"
    )
    print("vertical LS:", ordinary, "EIV:", popt)
    print(
        "ODR termination:",
        diagnostic.stopreason,
        "weighted SSE:",
        diagnostic.weighted_sse,
    )
    print("exact-time mean bands:", lower, upper)


WORKFLOWS: dict[str, Callable[[], None]] = {
    "confidence": confidence,
    "intervals": intervals,
    "diagnostics": diagnostics,
    "summary": summary,
    "profile": profile,
    "bootstrap": bootstrap,
    "collinearity": collinearity,
    "bounds": bounds,
    "propagation": propagation,
    "global": global_fit,
    "eiv": eiv,
}


def main() -> None:
    """Select one workflow or execute all eleven in documented order.

    Returns
    -------
    None
        Print scientific results to standard output; expected warnings remain visible.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workflow", nargs="?", choices=["all", *WORKFLOWS], default="all"
    )
    selection = parser.parse_args().workflow
    chosen = (
        WORKFLOWS.items() if selection == "all" else [(selection, WORKFLOWS[selection])]
    )
    for name, workflow in chosen:
        print(f"\n{name.upper()}")
        workflow()


if __name__ == "__main__":
    main()

"""Certificate-based partial-identification bounds for semantic exposure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .core import AuditError


@dataclass(frozen=True)
class ExposureConstraint:
    """A verified cardinality constraint on exposed evaluation identities."""

    name: str
    item_indices: tuple[int, ...]
    lower: int
    upper: int
    witness_type: str

    def __post_init__(self) -> None:
        if not self.name:
            raise AuditError("Constraint name cannot be empty.")
        if not self.item_indices:
            raise AuditError("Constraint group cannot be empty.")
        if len(self.item_indices) != len(set(self.item_indices)):
            raise AuditError("Constraint group contains duplicate indices.")
        if any(index < 0 for index in self.item_indices):
            raise AuditError("Constraint indices cannot be negative.")
        if not 0 <= self.lower <= self.upper <= len(self.item_indices):
            raise AuditError("Constraint bounds are incoherent.")
        if not self.witness_type:
            raise AuditError("Constraint witness type cannot be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_indices": list(self.item_indices),
            "lower": self.lower,
            "name": self.name,
            "upper": self.upper,
            "witness_type": self.witness_type,
        }


def singleton_constraint(
    *,
    name: str,
    item_index: int,
    exposed: bool,
    witness_type: str,
) -> ExposureConstraint:
    value = int(exposed)
    return ExposureConstraint(
        name=name,
        item_indices=(item_index,),
        lower=value,
        upper=value,
        witness_type=witness_type,
    )


def grouped_seed_constraint(
    *,
    name: str,
    item_indices: Iterable[int],
    training_episode_count: int,
    witness_type: str = "grouped_seed_pigeonhole",
) -> ExposureConstraint:
    indices = tuple(item_indices)
    if training_episode_count < 0:
        raise AuditError("Training episode count cannot be negative.")
    lower = int(training_episode_count > 0)
    upper = min(len(indices), training_episode_count)
    return ExposureConstraint(
        name=name,
        item_indices=indices,
        lower=lower,
        upper=upper,
        witness_type=witness_type,
    )


def solve_disjoint_certificate_bounds(
    *,
    item_count: int,
    constraints: Iterable[ExposureConstraint],
) -> dict[str, Any]:
    """Return sharp bounds relative to pairwise-disjoint certificates."""

    if item_count < 0:
        raise AuditError("Item count cannot be negative.")
    constraint_list = list(constraints)
    covered: set[int] = set()
    for constraint in constraint_list:
        group = set(constraint.item_indices)
        if any(index >= item_count for index in group):
            raise AuditError("Constraint index is outside the evaluation set.")
        overlap = covered & group
        if overlap:
            raise AuditError(
                f"Certificate groups are not disjoint: {sorted(overlap)}"
            )
        covered.update(group)

    uncovered_count = item_count - len(covered)
    lower = sum(constraint.lower for constraint in constraint_list)
    upper = (
        sum(constraint.upper for constraint in constraint_list)
        + uncovered_count
    )
    return {
        "certificate_count": len(constraint_list),
        "constraints": [
            constraint.to_dict() for constraint in constraint_list
        ],
        "exposure_count_lower_bound": lower,
        "exposure_count_upper_bound": upper,
        "exposure_prevalence_interval": {
            "denominator": item_count,
            "lower_numerator": lower,
            "upper_numerator": upper,
        },
        "item_count": item_count,
        "sharp_relative_to_certificate_system": True,
        "uncovered_count": uncovered_count,
    }


def assignment_satisfies_constraints(
    assignment: Iterable[bool | int],
    constraints: Iterable[ExposureConstraint],
) -> bool:
    values = [int(value) for value in assignment]
    if any(value not in (0, 1) for value in values):
        raise AuditError("Exposure assignment must be binary.")
    for constraint in constraints:
        if any(index >= len(values) for index in constraint.item_indices):
            raise AuditError("Constraint index is outside the assignment.")
        count = sum(values[index] for index in constraint.item_indices)
        if not constraint.lower <= count <= constraint.upper:
            return False
    return True


def is_laminar(groups: Iterable[Iterable[int]]) -> bool:
    """Return whether every pair of groups is disjoint or nested."""

    normalized = [set(group) for group in groups]
    for left_index, left in enumerate(normalized):
        for right in normalized[left_index + 1 :]:
            intersection = left & right
            if intersection and not (left <= right or right <= left):
                return False
    return True


def _overlap_matrix(
    item_count: int,
    constraints: list[ExposureConstraint],
) -> tuple[Any, Any, Any]:
    import numpy as np

    matrix = np.zeros((len(constraints), item_count), dtype=float)
    for row_index, constraint in enumerate(constraints):
        matrix[row_index, list(constraint.item_indices)] = 1.0
    lower = np.asarray([constraint.lower for constraint in constraints], dtype=float)
    upper = np.asarray([constraint.upper for constraint in constraints], dtype=float)
    return matrix, lower, upper


def _validate_overlap_inputs(
    item_count: int,
    constraints: Iterable[ExposureConstraint],
    weights: Iterable[float] | None,
) -> tuple[list[ExposureConstraint], Any]:
    import numpy as np

    if item_count < 0:
        raise AuditError("Item count cannot be negative.")
    constraint_list = list(constraints)
    for constraint in constraint_list:
        if any(index >= item_count for index in constraint.item_indices):
            raise AuditError("Constraint index is outside the evaluation set.")
    weight_array = (
        np.ones(item_count, dtype=float)
        if weights is None
        else np.asarray(list(weights), dtype=float)
    )
    if weight_array.shape != (item_count,):
        raise AuditError("Weights must contain one value per item.")
    if not np.isfinite(weight_array).all() or (weight_array < 0).any():
        raise AuditError("Weights must be finite and nonnegative.")
    return constraint_list, weight_array


def verify_dual_certificate(
    A: Any,
    lower: Any,
    upper: Any,
    weights: Any,
    alpha: Any,
    beta: Any,
    gamma: Any,
    claimed_bound: float,
    sense: str,
) -> bool:
    """Independently verify one LP weak-duality certificate."""

    import numpy as np

    if sense not in {"lower", "upper"}:
        raise AuditError("sense must be 'lower' or 'upper'.")
    matrix = np.asarray(A, dtype=float)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    objective = np.asarray(weights, dtype=float)
    alpha_array = np.asarray(alpha, dtype=float)
    beta_array = np.asarray(beta, dtype=float)
    gamma_array = np.asarray(gamma, dtype=float)
    if (
        matrix.ndim != 2
        or lower_array.shape != (matrix.shape[0],)
        or upper_array.shape != (matrix.shape[0],)
        or objective.shape != (matrix.shape[1],)
        or alpha_array.shape != (matrix.shape[0],)
        or beta_array.shape != (matrix.shape[0],)
        or gamma_array.shape != (matrix.shape[1],)
    ):
        return False
    values = (
        matrix,
        lower_array,
        upper_array,
        objective,
        alpha_array,
        beta_array,
        gamma_array,
    )
    if not all(np.isfinite(value).all() for value in values):
        return False
    tolerance = 1e-9
    if (
        (alpha_array < -tolerance).any()
        or (beta_array < -tolerance).any()
        or (gamma_array < -tolerance).any()
    ):
        return False
    signed_objective = objective if sense == "lower" else -objective
    dual_left = matrix.T @ (beta_array - alpha_array) - gamma_array
    if (dual_left > signed_objective + tolerance).any():
        return False
    dual_value = float(
        lower_array @ beta_array
        - upper_array @ alpha_array
        - gamma_array.sum()
    )
    if sense == "lower":
        return dual_value >= float(claimed_bound) - tolerance
    return -dual_value <= float(claimed_bound) + tolerance


def solve_overlapping_certificate_bounds(
    *,
    item_count: int,
    constraints: Iterable[ExposureConstraint],
    weights: Iterable[float] | None = None,
    method: str = "auto",
) -> dict[str, Any]:
    """Solve exact MILP bounds or dual-certified LP outer bounds."""

    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp

    if method not in {"auto", "milp", "lp"}:
        raise AuditError("method must be 'auto', 'milp', or 'lp'.")
    constraint_list, weight_array = _validate_overlap_inputs(
        item_count,
        constraints,
        weights,
    )
    matrix, lower, upper = _overlap_matrix(item_count, constraint_list)
    laminar = is_laminar(
        constraint.item_indices for constraint in constraint_list
    )
    selected_method = (
        ("milp" if item_count <= 200 else "lp")
        if method == "auto"
        else method
    )
    common = {
        "constraint_count": len(constraint_list),
        "laminar": laminar,
        "method": selected_method,
    }

    if selected_method == "milp":
        result: dict[str, Any] = {}
        linear_constraints = (
            [LinearConstraint(matrix, lower, upper)]
            if constraint_list
            else []
        )
        for sense, sign in (("lower", 1.0), ("upper", -1.0)):
            # HiGHS presolve can report an infeasible integer system as solved
            # (returning a point that violates an equality row), so the MILP
            # runs with presolve off and every returned point is checked
            # against the certificate rows before it is trusted.
            solution = milp(
                sign * weight_array,
                integrality=np.ones(item_count, dtype=np.int8),
                bounds=Bounds(0.0, 1.0),
                constraints=linear_constraints,
                options={"presolve": False},
            )
            if solution.status == 2:
                return {
                    **common,
                    "feasible": False,
                    "exact": True,
                    "attained_relative_to_certificate_system": False,
                }
            if solution.status != 0:
                raise AuditError(
                    f"MILP solver failed with status {solution.status}: "
                    f"{solution.message}"
                )
            if constraint_list:
                residuals = matrix @ np.rint(solution.x)
                if (
                    np.any(residuals < lower - 1e-6)
                    or np.any(residuals > upper + 1e-6)
                ):
                    raise AuditError(
                        "MILP returned a point that violates its "
                        "certificate rows."
                    )
            result[sense] = float(sign * solution.fun)
        return {
            **common,
            **result,
            "feasible": True,
            "exact": True,
            "attained_relative_to_certificate_system": True,
        }

    upper_matrix = (
        np.vstack([matrix, -matrix])
        if constraint_list
        else None
    )
    upper_vector = (
        np.concatenate([upper, -lower])
        if constraint_list
        else None
    )
    result = {}
    certificates: dict[str, dict[str, list[float]]] = {}
    for sense, sign in (("lower", 1.0), ("upper", -1.0)):
        solution = linprog(
            sign * weight_array,
            A_ub=upper_matrix,
            b_ub=upper_vector,
            bounds=[(0.0, 1.0)] * item_count,
            method="highs",
        )
        if solution.status == 2:
            return {
                **common,
                "feasible": False,
                "exact": False,
                "attained_relative_to_certificate_system": False,
            }
        if solution.status != 0:
            raise AuditError(
                f"LP solver failed with status {solution.status}: "
                f"{solution.message}"
            )
        result[sense] = float(sign * solution.fun)
        if constraint_list:
            multipliers = -np.asarray(
                solution.ineqlin.marginals,
                dtype=float,
            )
            alpha = multipliers[: len(constraint_list)]
            beta = multipliers[len(constraint_list) :]
        else:
            alpha = np.zeros(0, dtype=float)
            beta = np.zeros(0, dtype=float)
        gamma = -np.asarray(solution.upper.marginals, dtype=float)
        certificate = {
            "alpha": alpha.tolist(),
            "beta": beta.tolist(),
            "gamma": gamma.tolist(),
        }
        if not verify_dual_certificate(
            matrix,
            lower,
            upper,
            weight_array,
            certificate["alpha"],
            certificate["beta"],
            certificate["gamma"],
            result[sense],
            sense,
        ):
            raise AuditError(f"Generated {sense} LP certificate did not verify.")
        certificates[sense] = certificate
    return {
        **common,
        **result,
        "feasible": True,
        "exact": False,
        "attained_relative_to_certificate_system": False,
        "lp_dual_certificate": certificates,
    }

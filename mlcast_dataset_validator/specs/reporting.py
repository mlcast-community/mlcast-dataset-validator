from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple
from unittest import mock

from loguru import logger
from rich.console import Console
from rich.table import Table

# -------------------------
# Logging decorator and registry
# -------------------------
CHECK_REGISTRY: Dict[Tuple[str, str], Callable] = {}


def log_function_call(func):
    """Decorator to log function calls with their arguments.

    The original callable is stored in ``CHECK_REGISTRY`` so that it can be
    monkey patched (e.g., when printing specs without running validations).
    """

    if (func.__module__, func.__name__) not in CHECK_REGISTRY:
        CHECK_REGISTRY[(func.__module__, func.__name__)] = func

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Applying {func.__name__} with {kwargs}")
        func_from_registry = CHECK_REGISTRY[(func.__module__, func.__name__)]
        report = func_from_registry(*args, **kwargs)
        for result in report.results:
            result.module = func.__module__
            result.function = func.__name__
        return report

    return wrapper


# -------------------------
# Data structures
# -------------------------
@dataclass
class Result:
    section: str
    requirement: str
    status: str  # "FAIL" | "WARNING" | "PASS"
    detail: str = ""
    module: Optional[str] = None
    function: Optional[str] = None

    def __post_init__(self):
        valid_levels = {"FAIL", "WARNING", "PASS"}
        if self.status not in valid_levels:
            raise ValueError(
                f"Invalid status: {self.status}. Valid levels are: {', '.join(valid_levels)}."
            )


@dataclass
class ValidationReport:
    ok: bool = True
    results: List[Result] = field(default_factory=list)

    def add(
        self, section: str, requirement: str, status: str, detail: str = ""
    ) -> None:
        """
        Add a result to the validation report.

        Args:
            section (str): The section where the result occurred.
            requirement (str): The specific requirement that was evaluated.
            status (str): The severity status of the result ("FAIL", "WARNING", "INFO", "PASS").
            detail (str, optional): Additional details about the result. Defaults to an empty string.

        Returns:
            None
        """
        self.results.append(Result(section, requirement, status, detail))

    def summarize(self) -> str:
        """
        Summarize the validation report by counting results of each severity level.

        Returns:
            str: A summary string with counts of fails, warnings, and passes.
        """
        fails = sum(1 for r in self.results if r.status == "FAIL")
        warns = sum(1 for r in self.results if r.status == "WARNING")
        passes = sum(1 for r in self.results if r.status == "PASS")
        return f"Summary: {fails} fail(s), {warns} warning(s), {passes} pass(es)."

    def __iadd__(self, other: "ValidationReport") -> "ValidationReport":
        """
        Merge another ValidationReport into this one (in-place).

        Args:
            other (ValidationReport): The other validation report to merge.

        Returns:
            ValidationReport: The updated validation report (self).
        """
        self.results.extend(other.results)
        self.ok = self.ok and other.ok
        return self

    def __add__(self, other: "ValidationReport") -> "ValidationReport":
        """
        Combine two ValidationReports into a new one.

        Args:
            other (ValidationReport): The other validation report to combine.

        Returns:
            ValidationReport: A new validation report containing results from both.
        """
        out = ValidationReport(ok=self.ok and other.ok)
        out.results = [*self.results, *other.results]
        return out

    def console_print(self, *, file=None) -> None:
        """
        Print all results in the validation report as a table using the rich library.

        Args:
            file: Optional file-like object to write to (defaults to stdout).

        Returns:
            None
        """
        console = Console(file=file)
        table = Table(title="Validation Report")

        table.add_column("Section", style="bold")
        table.add_column("Requirement", style="dim")
        table.add_column("Status", justify="center")
        table.add_column("Detail", style="italic")
        table.add_column(
            "Checking function", style="bold"
        )  # New column for function name

        level_emojis = {"FAIL": "❌", "WARNING": "⚠️", "PASS": "✅"}

        for result in self.results:
            if result.module and result.function:
                fn_fqn = f"{result.module}.{result.function}".removeprefix(
                    "mlcast_dataset_validator.checks."
                )
            else:
                fn_fqn = "N/A"
            table.add_row(
                result.section,
                result.requirement,
                level_emojis.get(result.status, result.status),
                result.detail,
                fn_fqn,
            )

        console.print(table)
        console.print(self.summarize())

    def has_fails(self) -> bool:
        """
        Check if the report contains any FAIL results.

        Returns:
            bool: True if there is at least one FAIL result, False otherwise.
        """
        return any(r.status == "FAIL" for r in self.results)

    def has_warnings(self) -> bool:
        """
        Check if the report contains any WARNING results.

        Returns:
            bool: True if there is at least one WARNING result, False otherwise.
        """
        return any(r.status == "WARNING" for r in self.results)


@contextmanager
def skip_all_checks():
    """Context manager to bypass check functions and dataset loading.

    Assumes check functions are decorated with ``log_function_call`` which
    records them in ``CHECK_REGISTRY``. Each registered check is monkey patched
    to a stub returning an empty ``ValidationReport`` and dataset loading is
    bypassed.
    """

    def _stubbed_check(*_args, **_kwargs):
        return ValidationReport()

    with ExitStack() as stack:
        # Avoid calling into real xarray when rendering specs.
        try:
            stack.enter_context(mock.patch("xarray.open_zarr", lambda *a, **kw: None))
            import xarray as xr

            stack.enter_context(
                mock.patch.object(xr, "open_zarr", lambda *a, **kw: None)
            )
        except ModuleNotFoundError:
            # xarray not available in this environment; skip patching dataset load.
            pass

        stack.enter_context(
            mock.patch.dict(
                CHECK_REGISTRY,
                {key: _stubbed_check for key in list(CHECK_REGISTRY.keys())},
            )
        )
        yield

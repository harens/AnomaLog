"""Utilities for building template sequences from structured log lines.

The module groups parsed log lines into windows (entity, fixed-size, or
time-based) and decorates them with inferred templates and anomaly labels.
"""

from __future__ import annotations

import functools
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

from typing_extensions import Self, override

from anomalog.parsers.structured.contracts import is_anomalous_label
from anomalog.representations import (
    SequenceRepresentation,
    SequenceRepresentationView,
    TRepresentation,
)
from anomalog.split_validation import validate_split_fractions

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterator

    from anomalog.parsers.structured.contracts import StructuredLine, StructuredSink
    from anomalog.parsers.template.dataset import (
        ExtractedParameters,
        LogTemplate,
        TemplatedDataset,
    )
    from experiments.models.base import SequenceSummary


class SplitLabel(str, Enum):
    """Dataset split membership for a sequence.

    Attributes:
        TRAIN: Sequence belongs to the training split.
        TEST: Sequence belongs to the evaluation/test split.
        IGNORED: Sequence belongs to the fixed train pool but is not used for
            the current training prefix.
    """

    TRAIN = "train"
    TEST = "test"
    IGNORED = "ignored"


@dataclass(slots=True, frozen=True)
class SequenceSplitSummary:
    """Serialisable summary of requested versus effective split behavior.

    The requested train fraction may not equal the effective one after
    grouping-specific eligibility rules are applied. Persisting both protects
    downstream experiment manifests from silently overstating how much data was
    actually available for training.

    Attributes:
        requested_train_fraction (float): Requested fraction provided by the
            caller.
        requested_test_fraction (float): Requested test suffix fraction
            provided by the caller.
        train_on_normal_entities_only (bool | None): Whether training was restricted to
            normal entities only. Only applicable to entity grouping; `None` otherwise.
        train_pool_sequence_count (int): Number of sequences in the
            chronological train candidate window before detector-specific
            filtering is applied.
        ineligible_train_pool_count (int): Number of sequences in the train
            pool that were ineligible for training under the current policy.
        realised_train_sequence_count (int): Number of sequences actually used
            for training after any detector-specific filtering.
        excluded_from_train_count (int): Number of sequences withheld from the
            train pool before scoring, including the ignored middle band and
            any detector-ineligible prefix items.
        eligible_train_sequence_count (int): Number of sequences in the
            denominator for the effective train-fraction calculation. In
            entity-grouped mode this is the fixed chronological train pool, or
            the normal-only subset of that pool when normal-only training is
            enabled.
        ignored_sequence_count (int): Number of sequences withheld from the
            train pool because they fell outside the requested train prefix or
            were ineligible under the current filtering policy.
        effective_train_fraction_of_eligible (float): Realised train fraction
            over the eligible set.
        effective_train_fraction_overall (float): Realised train fraction over
            the full generated sequence population.
    """

    requested_train_fraction: float
    requested_test_fraction: float
    train_on_normal_entities_only: bool | None
    train_pool_sequence_count: int
    ineligible_train_pool_count: int
    realised_train_sequence_count: int
    excluded_from_train_count: int
    eligible_train_sequence_count: int
    ignored_sequence_count: int
    effective_train_fraction_of_eligible: float
    effective_train_fraction_overall: float

    def as_dict(self) -> dict[str, int | float | bool | str]:
        """Return a stable JSON-friendly representation.

        Returns:
            dict[str, int | float | bool | str]: Serialised split summary.
        """
        results: dict[str, int | float | bool | str] = {
            "requested_train_fraction": self.requested_train_fraction,
            "requested_test_fraction": self.requested_test_fraction,
            "train_pool_sequence_count": self.train_pool_sequence_count,
            "ineligible_train_pool_count": self.ineligible_train_pool_count,
            "realised_train_sequence_count": self.realised_train_sequence_count,
            "excluded_from_train_count": self.excluded_from_train_count,
            "eligible_train_sequence_count": self.eligible_train_sequence_count,
            "ignored_sequence_count": self.ignored_sequence_count,
            "effective_train_fraction_of_eligible": (
                self.effective_train_fraction_of_eligible
            ),
            "effective_train_fraction_overall": self.effective_train_fraction_overall,
        }

        if self.train_on_normal_entities_only is not None:
            results["train_on_normal_entities_only"] = (
                self.train_on_normal_entities_only
            )

        return results


@dataclass(slots=True, frozen=True)
class SequenceSplitCounts:
    """Exact split counts for a concrete sequence builder.

    Attributes:
        total_count (int): Total emitted sequence count.
        train_count (int): Count assigned to the current train prefix.
        ignored_count (int): Count withheld between train and test.
        test_count (int): Count assigned to the fixed test suffix.
    """

    total_count: int
    train_count: int
    ignored_count: int
    test_count: int


@dataclass(slots=True, frozen=True)
class TemplateSequence:
    """Grouped log window before any model-specific representation is applied.

    This keeps sequence semantics such as event ordering, labels, and entity
    membership. Model inputs derived from it live in `SequenceSample`.

    Attributes:
        events (list[tuple[str, list[str], int | None]]): Ordered sequence events
            as `(template, parameters, dt_prev_ms)` tuples.
        label (int): Sequence-level anomaly label derived from rows and group
            labels.
        entity_ids (list[str]): Unique entity ids present in the window in
            first-seen order.
        window_id (int): Stable window identifier assigned by the builder.
        split_label (SplitLabel): Dataset split assigned to the sequence.
    """

    events: list[
        tuple[str, list[str], int | None]
    ]  # (template, parameters, dt_prev_ms)
    label: int
    entity_ids: list[str]  # unique entity ids present (may be empty)
    window_id: int
    split_label: SplitLabel = SplitLabel.TRAIN

    @property
    def templates(self) -> list[str]:
        """Return the ordered template strings for this sequence."""
        return [tpl for tpl, _, _ in self.events]

    @property
    def sole_entity_id(self) -> str | None:
        """Return the entity id when the sequence belongs to exactly one entity.

        If multiple entities appear in the window, None is returned to avoid
        implying a single owning entity.
        """
        if len(self.entity_ids) == 1:
            return self.entity_ids[0]
        return None


@dataclass(slots=True, frozen=True)
class SequenceBuilder(ABC, Iterable[TemplateSequence]):
    """Common sequence-building behavior shared across grouping strategies.

    Sequence builders stay lazy so expensive grouping, template inference, and
    label resolution only happen when a caller iterates. The shared base also
    centralises split assignment so experiment manifests can describe train/test
    semantics consistently across grouping modes.

    Attributes:
        sink (StructuredSink): Structured sink supplying grouped rows.
        infer_template (Callable[[str], tuple[LogTemplate, ExtractedParameters]]):
            Template inference function for row message text.
        label_for_group (Callable[[str], int | None]): Group-level anomaly label
            lookup by entity id.
        train_frac (float): Requested training fraction for the builder.
        test_frac (float): Fixed test suffix fraction.
    """

    sink: StructuredSink
    infer_template: Callable[[str], tuple[LogTemplate, ExtractedParameters]]
    label_for_group: Callable[[str], int | None]
    train_frac: float = 0.2
    test_frac: float = 0.8

    def __post_init__(self) -> None:
        """Validate the requested split fractions."""
        validate_split_fractions(
            train_frac=self.train_frac,
            test_frac=self.test_frac,
        )

    def _split_counts(self, total_count: int) -> SequenceSplitCounts:
        """Return train, ignored, and test counts for one chronological split.

        Args:
            total_count (int): Total emitted sequence count.

        Returns:
            SequenceSplitCounts: Exact split counts for the requested split
                configuration.
        """
        if total_count <= 0:
            return SequenceSplitCounts(
                total_count=0,
                train_count=0,
                ignored_count=0,
                test_count=0,
            )
        test_count = min(total_count, math.ceil(self.test_frac * total_count))
        train_count = min(
            total_count - test_count,
            math.ceil(self.train_frac * total_count),
        )
        ignored_count = total_count - train_count - test_count
        return SequenceSplitCounts(
            total_count=total_count,
            train_count=train_count,
            ignored_count=ignored_count,
            test_count=test_count,
        )

    def train_fraction_eligible_sequence_count(
        self,
        *,
        sequence_summary: SequenceSummary,
    ) -> int:
        """Return the denominator for effective train-fraction accounting.

        Args:
            sequence_summary (SequenceSummary): Aggregate split and label counts.

        Returns:
            int: Count of sequences considered eligible when reporting the
                realised train fraction for this grouping strategy. This is not
                necessarily the number of sequences that were actually assigned
                to train.
        """
        if not self.test_frac:
            return sequence_summary.sequence_count
        return sequence_summary.sequence_count - sequence_summary.test_sequence_count

    def split_count_hint(self) -> SequenceSplitCounts | None:
        """Return a cheap exact split-count summary when the builder knows it.

        Returns:
            SequenceSplitCounts | None: Exact split counts when cheaply
                available, otherwise `None`.
        """
        del self
        return None

    def with_split_fractions(
        self,
        train_frac: float,
        test_frac: float,
    ) -> Self:
        """Return a copy with both split fractions updated together.

        Args:
            train_frac (float): Requested fraction of the total population to
                assign to the train prefix.
            test_frac (float): Requested fraction reserved for the fixed test
                suffix.

        Returns:
            Self: Copy with updated split fractions.
        """
        return replace(self, train_frac=train_frac, test_frac=test_frac)

    def represent_with(
        self,
        representation: SequenceRepresentation[TRepresentation],
    ) -> SequenceRepresentationView[TRepresentation]:
        """Return a lazy builder that applies a representation per sequence.

        Args:
            representation (SequenceRepresentation[TRepresentation]): Sequence
                representation to apply lazily to each built sequence.

        Returns:
            SequenceRepresentationView[TRepresentation]: Lazy represented view of
                the generated sequences.
        """
        return SequenceRepresentationView(sequences=self, representation=representation)

    def train_sequence_count_unit_hint(self) -> str | None:
        """Return a human-readable unit label for train-count progress.

        This is intended for progress reporting only. Builders should return a
        unit when it clarifies what the bounded train count represents, such as
        ``"entities"`` for entity grouping.

        Returns:
            str | None: Unit label for train-count progress when useful,
                otherwise ``None``.
        """
        del self
        return None

    def split_summary_train_on_normal_entities_only(self) -> bool | None:
        """Return split-summary metadata for entity-only normal training.

        Returns:
            bool | None: Whether train was restricted to normal entities only,
                or `None` when that concept does not apply to this builder.
        """
        del self
        return None

    @abstractmethod
    def __iter__(self) -> Iterator[TemplateSequence]:
        """Iterate over template sequences yielded by the configured grouping.

        Returns:
            Iterator[TemplateSequence]: Iterator yielding grouped and
                template-enriched sequences.
        """
        ...

    def build_split_summary(
        self,
        *,
        sequence_summary: SequenceSummary,
    ) -> SequenceSplitSummary:
        """Describe requested versus effective split semantics for one run.

        Args:
            sequence_summary (SequenceSummary): Aggregate split and label counts.

        Returns:
            SequenceSplitSummary: Requested and effective split metrics.
        """
        eligible_train_sequence_count = self.train_fraction_eligible_sequence_count(
            sequence_summary=sequence_summary,
        )
        realised_train_sequence_count = sequence_summary.train_sequence_count
        train_pool_sequence_count = (
            realised_train_sequence_count + sequence_summary.ignored_sequence_count
        )
        ineligible_train_pool_count = (
            sum(
                count
                for label, count in sequence_summary.ignored_label_counts.items()
                if is_anomalous_label(label)
            )
            if self.split_summary_train_on_normal_entities_only()
            else 0
        )
        excluded_from_train_count = (
            train_pool_sequence_count - realised_train_sequence_count
        )
        effective_train_fraction_of_eligible = (
            realised_train_sequence_count / eligible_train_sequence_count
            if eligible_train_sequence_count
            else 0.0
        )
        effective_train_fraction_overall = (
            realised_train_sequence_count / sequence_summary.sequence_count
            if sequence_summary.sequence_count
            else 0.0
        )
        return SequenceSplitSummary(
            requested_train_fraction=self.train_frac,
            requested_test_fraction=self.test_frac,
            train_on_normal_entities_only=(
                self.split_summary_train_on_normal_entities_only()
            ),
            train_pool_sequence_count=train_pool_sequence_count,
            ineligible_train_pool_count=ineligible_train_pool_count,
            realised_train_sequence_count=realised_train_sequence_count,
            excluded_from_train_count=excluded_from_train_count,
            eligible_train_sequence_count=eligible_train_sequence_count,
            ignored_sequence_count=sequence_summary.ignored_sequence_count,
            effective_train_fraction_of_eligible=round(
                effective_train_fraction_of_eligible,
                8,
            ),
            effective_train_fraction_overall=round(
                effective_train_fraction_overall,
                8,
            ),
        )

    @abstractmethod
    def iter_grouped_rows(self) -> Iterator[Collection[StructuredLine]]:
        """Return grouped rows for the configured strategy.

        Returns:
            Iterator[Collection[StructuredLine]]: Iterator over grouped windows
                of structured rows.
        """
        ...

    def _build_sequence(
        self,
        window_id: int,
        rows: Collection[StructuredLine],
        infer_template: Callable[[str], tuple[LogTemplate, ExtractedParameters]],
        label_for_group: Callable[[str], int | None],
        split_label: SplitLabel,
    ) -> TemplateSequence | None:
        """Convert a non-empty row window into a labelled template sequence.

        Args:
            window_id (int): Monotonic window identifier for the generated
                sequence.
            rows (Collection[StructuredLine]): Structured rows in the current
                window.
            infer_template (Callable[[str], tuple[LogTemplate, ExtractedParameters]]):
                Template inference function for untemplated row text.
            label_for_group (Callable[[str], int | None]): Group-level anomaly
                label lookup by entity id.
            split_label (SplitLabel): Assigned dataset split for the sequence.

        Returns:
            TemplateSequence | None: Built sequence, or `None` for empty
                windows. Sequence labels are derived from both inline row
                labels and resolved group labels under the shared anomaly
                semantics.
        """
        if not rows:
            return None

        events: list[tuple[str, list[str], int | None]] = []
        seq_label = 0
        prev_ts: int | None = None

        unique_ids = self._entity_ids_for_rows(rows)

        for r in rows:
            template, params = infer_template(r.untemplated_message_text)
            dt, prev_ts = self._compute_dt(prev_ts, r.timestamp_unix_ms)

            events.append((template, list(params), dt))

            if seq_label == 1:
                continue

            line_lab = getattr(r, "anomalous", None)
            if is_anomalous_label(line_lab):
                seq_label = 1
                continue

            ent = r.entity_id
            if ent is not None and is_anomalous_label(label_for_group(ent)):
                seq_label = 1

        return TemplateSequence(
            events=events,
            label=seq_label,
            entity_ids=unique_ids,
            window_id=window_id,
            split_label=split_label,
        )

    def _entity_ids_for_rows(self, rows: Collection[StructuredLine]) -> list[str]:
        """Return unique entity ids for one window in first-seen order.

        Args:
            rows (Collection[StructuredLine]): Structured rows belonging to one
                grouped window.

        Returns:
            list[str]: Unique entity ids in first-seen order.
        """
        del self
        seen: set[str] = set()
        entity_ids: list[str] = []
        for row in rows:
            if row.entity_id is None or row.entity_id in seen:
                continue
            seen.add(row.entity_id)
            entity_ids.append(row.entity_id)
        return entity_ids

    @staticmethod
    def _compute_dt(
        prev_ts: int | None,
        ts: int | None,
    ) -> tuple[int | None, int | None]:
        """Compute delta time between events while preserving previous ts.

        Args:
            prev_ts (int | None): Previous event timestamp in milliseconds.
            ts (int | None): Current event timestamp in milliseconds.

        Examples:
            >>> SequenceBuilder._compute_dt(None, 1000)
            (None, 1000)
            >>> SequenceBuilder._compute_dt(1000, 1250)
            (250, 1250)
            >>> SequenceBuilder._compute_dt(2000, None)
            (None, 2000)

        Returns:
            tuple[int | None, int | None]: Delta from the previous timestamp and
                the updated previous timestamp to carry forward.
        """
        if ts is None:
            return None, prev_ts
        if prev_ts is None:
            return None, ts
        return int(ts) - int(prev_ts), ts

    @staticmethod
    def _count_fixed_windows(
        *,
        sink: StructuredSink,
        window_size: int,
        step: int | None,
    ) -> int:
        """Estimate number of fixed windows given window and step sizes.

        Args:
            sink (StructuredSink): Structured sink providing the total row count.
            window_size (int): Number of rows per window.
            step (int | None): Step between successive windows, or `None` to use
                `window_size`.

        Examples:
            >>> class _Sink:
            ...     def count_rows(self):
            ...         return 10
            ...
            >>> sb = FixedSequenceBuilder(
            ...     sink=_Sink(),
            ...     infer_template=lambda s: (s, ()),
            ...     label_for_group=lambda _: 0,
            ...     window_size=4,
            ...     step=2,
            ... )
            >>> sb.count_windows()
            4

        Returns:
            int: Estimated count of fixed windows for the sink.
        """
        if window_size <= 0:
            return 0
        step = step or window_size
        if step <= 0:
            return 0
        n = sink.count_rows()
        if n <= 0:
            return 0

        if n <= window_size:
            return 1
        return 1 + math.ceil((n - window_size) / step)

    @staticmethod
    def _count_time_windows(
        *,
        sink: StructuredSink,
        time_span_ms: int,
        step: int | None,
    ) -> int:
        """Estimate number of time windows from sink timestamp bounds.

        Args:
            sink (StructuredSink): Structured sink providing timestamp bounds.
            time_span_ms (int): Width of each time window in milliseconds.
            step (int | None): Step between successive windows, or `None` to use
                `time_span_ms`.

        Examples:
            >>> class _Sink:
            ...     def timestamp_bounds(self):
            ...         return 1_000, 3_500
            ...
            >>> sb = TimeSequenceBuilder(
            ...     sink=_Sink(),
            ...     infer_template=lambda s: (s, ()),
            ...     label_for_group=lambda _: 0,
            ...     time_span_ms=1_000,
            ...     step=500,
            ... )
            >>> sb.count_windows()
            4

        Returns:
            int: Estimated count of time windows for the sink.
        """
        if time_span_ms <= 0:
            return 0
        step = step or time_span_ms
        if step <= 0:
            return 0
        first_ts, last_ts = sink.timestamp_bounds()
        if first_ts is None or last_ts is None:
            return 0
        if last_ts < first_ts:
            return 0
        span = time_span_ms
        duration = last_ts - first_ts

        if duration < span:
            return 1

        return (duration - span) // step + 1


@dataclass(slots=True, frozen=True, kw_only=True)
class EntitySequenceBuilder(SequenceBuilder):
    """Sequence builder for per-entity grouping.

    Attributes:
        train_on_normal_entities_only (bool): Whether anomalous entities are
            excluded from the training split budget.
    """

    train_on_normal_entities_only: bool = False

    @classmethod
    def from_dataset(
        cls,
        td: TemplatedDataset,
    ) -> Self:
        """Create an entity-grouped builder from a templated dataset.

        Args:
            td (TemplatedDataset): Templated dataset to bind into the builder.

        Returns:
            Self: Builder bound to the templated dataset.
        """
        return cls(
            sink=td.sink,
            infer_template=td.template_parser.inference,
            label_for_group=td.anomaly_labels.label_for_group,
        )

    def with_train_on_normal_entities_only(
        self,
        *,
        enabled: bool = True,
    ) -> Self:
        """Limit training sequences to entities without anomalies.

        Args:
            enabled (bool): Whether to restrict train sequences to normal
                entities only.

        Returns:
            Self: Copy with updated normal-only training behavior.
        """
        return replace(self, train_on_normal_entities_only=enabled)

    def _entity_split_counts(
        self,
        *,
        label_for_group: Callable[[str], int | None],
    ) -> tuple[SequenceSplitCounts, int]:
        """Return cached-free split counts for the fixed entity chronology.

        Args:
            label_for_group (Callable[[str], int | None]): Entity label lookup
                used to determine whether a group is anomalous.

        Returns:
            tuple[SequenceSplitCounts, int]: Exact split counts and eligible
                normal count within the train pool.
        """
        entity_counts = self.sink.count_entities_by_label(label_for_group)
        total_entities = entity_counts.total_entities
        counts = self._split_counts(total_entities)
        train_pool_count = total_entities - counts.test_count
        normal_pool_count = train_pool_count
        if self.train_on_normal_entities_only:
            normal_pool_count = 0
            for index, rows in enumerate(self.iter_grouped_rows()):
                if index >= train_pool_count:
                    break
                entity_id = next(
                    (row.entity_id for row in rows if row.entity_id is not None),
                    None,
                )
                if entity_id is None:
                    continue
                if not is_anomalous_label(label_for_group(entity_id)):
                    normal_pool_count += 1
        return counts, normal_pool_count

    @override
    def train_fraction_eligible_sequence_count(
        self,
        *,
        sequence_summary: SequenceSummary,
    ) -> int:
        """Return the denominator for effective train-fraction accounting.

        Args:
            sequence_summary (SequenceSummary): Aggregate split and label counts.

        Returns:
            int: Eligible entity-sequence count under the current policy. When
                normal-only training is enabled, this counts only normal
                entities in the fixed train pool before the hold-out suffix.
        """
        if not self.train_on_normal_entities_only:
            return (
                sequence_summary.sequence_count - sequence_summary.test_sequence_count
            )
        normal_train_count = sum(
            count
            for label, count in sequence_summary.train_label_counts.items()
            if not is_anomalous_label(label)
        )
        normal_train_count += sum(
            count
            for label, count in sequence_summary.ignored_label_counts.items()
            if not is_anomalous_label(label)
        )
        if normal_train_count:
            return normal_train_count
        return 0

    @override
    def split_count_hint(self) -> SequenceSplitCounts:
        """Return the exact split-count summary for entity grouping.

        Returns:
            SequenceSplitCounts: Exact split counts for the entity builder.
        """
        counts, _ = self._entity_split_counts(label_for_group=self.label_for_group)
        return counts

    @override
    def train_sequence_count_unit_hint(self) -> str:
        """Return the unit label for entity-grouped train progress.

        Returns:
            str: Unit label for entity-grouped train progress.
        """
        return "entities"

    @override
    def split_summary_train_on_normal_entities_only(self) -> bool:
        """Return entity split-summary metadata for normal-only training.

        Returns:
            bool: Whether train was restricted to normal entities only.
        """
        return self.train_on_normal_entities_only

    @override
    def iter_grouped_rows(self) -> Iterator[Collection[StructuredLine]]:
        """Return rows grouped by entity.

        Returns:
            Iterator[Collection[StructuredLine]]: Entity-grouped structured rows.
        """
        return self.sink.iter_entity_sequences()()

    @override
    def _entity_ids_for_rows(self, rows: Collection[StructuredLine]) -> list[str]:
        """Return the single entity id for an entity-grouped window.

        Args:
            rows (Collection[StructuredLine]): Structured rows belonging to one
                entity-grouped window.

        Returns:
            list[str]: Single entity id when present, otherwise an empty list.
        """
        for row in rows:
            if row.entity_id is not None:
                return [row.entity_id]
        return []

    def __iter__(self) -> Iterator[TemplateSequence]:
        """Iterate over template sequences yielded by the configured grouping.

        Yields:
            TemplateSequence: One grouped and template-enriched sequence.
        """
        infer_template = functools.lru_cache(maxsize=50_000)(self.infer_template)
        label_for_group = functools.lru_cache(maxsize=100_000)(self.label_for_group)
        counts, _ = self._entity_split_counts(label_for_group=label_for_group)
        normals_seen_in_train = 0
        test_start_index = counts.total_count - counts.test_count

        for window_id, rows in enumerate(self.iter_grouped_rows()):
            entity_id = next(
                (row.entity_id for row in rows if row.entity_id is not None),
                None,
            )
            entity_is_anomalous = entity_id is not None and is_anomalous_label(
                label_for_group(entity_id),
            )
            if window_id >= test_start_index:
                split_label = SplitLabel.TEST
            elif self.train_on_normal_entities_only:
                split_label = (
                    SplitLabel.TRAIN
                    if (not entity_is_anomalous)
                    and (normals_seen_in_train < counts.train_count)
                    else SplitLabel.IGNORED
                )
            else:
                split_label = (
                    SplitLabel.TRAIN
                    if window_id < counts.train_count
                    else SplitLabel.IGNORED
                )

            seq = self._build_sequence(
                window_id,
                rows,
                infer_template,
                label_for_group,
                split_label,
            )
            if seq is not None:
                if (
                    self.train_on_normal_entities_only
                    and split_label is SplitLabel.TRAIN
                    and seq.label == 0
                ):
                    normals_seen_in_train += 1
                yield seq


@dataclass(slots=True, frozen=True, kw_only=True)
class NonEntitySequenceBuilder(SequenceBuilder):
    """Sequence builder for non-entity grouping strategies.

    This is a marker subclass to clarify when normal entity logic
    does not apply, such as for fixed-size or time-based windowing.
    """

    @abstractmethod
    def count_windows(self) -> int:
        """Return the total number of windows implied by the sink and config.

        Returns:
            int: Count of windows implied by the sink and current builder config.
        """
        ...

    @override
    def split_count_hint(self) -> SequenceSplitCounts:
        """Return the exact split-count summary for non-entity grouping.

        Returns:
            SequenceSplitCounts: Exact split counts for the grouping strategy.
        """
        return self._split_counts(self.count_windows())

    def __iter__(self) -> Iterator[TemplateSequence]:
        """Iterate over template sequences yielded by the configured grouping.

        Yields:
            TemplateSequence: One grouped and template-enriched sequence.
        """
        # Non-entity grouping: simple positional cutoff
        rows_iter = self.iter_grouped_rows()
        infer_template = functools.lru_cache(maxsize=50_000)(self.infer_template)
        label_for_group = functools.lru_cache(maxsize=100_000)(self.label_for_group)

        split_counts = self.split_count_hint()
        train_limit = split_counts.train_count
        test_start = split_counts.total_count - split_counts.test_count

        for window_id, rows in enumerate(rows_iter):
            if window_id >= test_start:
                split_label = SplitLabel.TEST
            elif window_id < train_limit:
                split_label = SplitLabel.TRAIN
            else:
                split_label = SplitLabel.IGNORED
            seq = self._build_sequence(
                window_id,
                rows,
                infer_template,
                label_for_group,
                split_label,
            )
            if seq is not None:
                yield seq


@dataclass(slots=True, frozen=True, kw_only=True)
class FixedSequenceBuilder(NonEntitySequenceBuilder):
    """Sequence builder for fixed-size window grouping.

    Attributes:
        window_size (int): Number of rows per emitted window.
        step (int | None): Row advance between windows. `None` means
            non-overlapping windows.
    """

    window_size: int
    step: int | None = None

    @override
    def iter_grouped_rows(self) -> Iterator[Collection[StructuredLine]]:
        """Return rows grouped by fixed-size windows.

        Returns:
            Iterator[Collection[StructuredLine]]: Fixed-size row windows.
        """
        return self.sink.iter_fixed_window_sequences(
            self.window_size,
            step_size=self.step,
        )()

    @override
    def count_windows(self) -> int:
        """Return the number of fixed-size windows.

        Returns:
            int: Count of fixed-size windows implied by the sink and config.
        """
        return self._count_fixed_windows(
            sink=self.sink,
            window_size=self.window_size,
            step=self.step,
        )

    @override
    def train_sequence_count_unit_hint(self) -> str:
        """Return the unit label for fixed-window train progress.

        Returns:
            str: Unit label for fixed-window train progress.
        """
        return "windows"


@dataclass(slots=True, frozen=True, kw_only=True)
class TimeSequenceBuilder(NonEntitySequenceBuilder):
    """Sequence builder for time-window grouping.

    Attributes:
        time_span_ms (int): Duration of each emitted window in milliseconds.
        step (int | None): Window advance in milliseconds. `None` means
            non-overlapping windows.
    """

    time_span_ms: int
    step: int | None = None

    @override
    def train_sequence_count_unit_hint(self) -> str:
        """Return the unit label for time-window train progress.

        Returns:
            str: Unit label for time-window train progress.
        """
        return "windows"

    @override
    def iter_grouped_rows(self) -> Iterator[Collection[StructuredLine]]:
        """Return rows grouped by time windows.

        Returns:
            Iterator[Collection[StructuredLine]]: Time-based row windows.
        """
        return self.sink.iter_time_window_sequences(
            self.time_span_ms,
            step_span_ms=self.step,
        )()

    @override
    def count_windows(self) -> int:
        """Return the number of time windows.

        Returns:
            int: Count of time windows implied by the sink timestamps and config.
        """
        return self._count_time_windows(
            sink=self.sink,
            time_span_ms=self.time_span_ms,
            step=self.step,
        )

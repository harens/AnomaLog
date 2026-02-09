import re
from datetime import datetime

from dateutil.tz import UTC
from prefect.logging import get_logger

from anomalog.structured_parsers.contracts import StructuredLine, StructuredParser


class HDFSV1Parser(StructuredParser):
    pass  # TODO: Implement


class BGLParser(StructuredParser):
    # Matches both:
    #   - <epoch> <date> <loc> <hires_ts> <loc> <tail>
    #   <prefix> <epoch> <date> <loc> <hires_ts> <tail>
    #
    # with optional leading "-" that indicates "normal" in BGL.
    _BGL_RE = re.compile(
        r"""
        ^\s*
        (?P<dash>-)?\s*
        (?:(?P<prefix>\d+:\S+)\s+)?(?:\S+\s+)?
        (?P<epoch>\d+)\s+
        (?P<date>\d{4}\.\d{2}\.\d{2})\s+
        (?P<entity>\S+)\s+
        (?P<hires_ts>\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d+)\s+
        (?P<entity2>\S+)\s+
        (?P<tail>\S+\s+\S+\s+\S+.*)                      # FAC SUB SEV <rest...>
        \s*$
        """,
        re.VERBOSE,
    )

    @staticmethod
    def _hires_ts_to_unix_ms(ts: str) -> int | None:
        # BGL tooling usually treats these as UTC; adjust if you decide otherwise.
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d-%H.%M.%S.%f").replace(tzinfo=UTC)
            return int(dt.timestamp() * 1000)
        except ValueError:
            return None

    def parse_line(self, raw_line: str, line_order: int) -> StructuredLine | None:
        s = raw_line.rstrip("\n")
        logger = get_logger()

        m = BGLParser._BGL_RE.match(s)
        if not m:
            logger.warning("Cannot parse BGL line %d: %r", line_order, s)
            return None

        d = m.groupdict()

        anomalous = 0 if d["dash"] == "-" else 1
        entity_id = d["entity"]

        ts_ms = BGLParser._hires_ts_to_unix_ms(d["hires_ts"])
        if ts_ms is None:
            # Fallback to epoch seconds if needed
            logger.warning(
                "Failed to parse hires timestamp %r for raw line %r, "
                "falling back to epoch seconds.",
                d["hires_ts"],
                s,
            )
            try:
                ts_ms = int(d["epoch"]) * 1000
            except ValueError:
                ts_ms = None

        untemplated = d["tail"].strip()

        return StructuredLine(
            line_order=line_order,
            timestamp_unix_ms=ts_ms,
            entity_id=entity_id,
            untemplated_message_text=untemplated,
            anomalous=anomalous,
        )

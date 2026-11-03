"""Keep query strings out of the HTTP client library's request log (platform-wide).

httpx logs every request at INFO as `HTTP Request: GET <full URL> "HTTP/1.1 200
OK"`. For the EHR sync that URL is `…/Observation?patient=<FHIR patient id>&…`:
one line per search, about 13,000 per sync of a 700-patient practice (measured
on the SMART Health IT launcher, 8 Oct 2026), each naming a patient. The sync's
own logging already refuses query strings (business `adapters/fhir/sync.py` `_log_path`); this
closes the library's side.

A filter on the `httpx` logger rewrites the URL argument to drop its query. It
affects every httpx request this process logs, not only the EHR's: a query string
is where OAuth codes, API keys and identifiers travel, so none of them belongs in
a log line. Method, host, path and status stay, which is what diagnosis needs.
"""

from __future__ import annotations

import logging

_MARKER = "HTTP Request:"


class RedactHttpxQuery(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if (
            isinstance(record.msg, str)
            and record.msg.startswith(_MARKER)
            and isinstance(record.args, tuple)
            and len(record.args) >= 2
        ):
            url = str(record.args[1])
            if "?" in url:
                args = list(record.args)
                args[1] = url.split("?", 1)[0] + "?<redacted>"
                record.args = tuple(args)
        return True


def install() -> None:
    """Idempotent: one filter on the httpx logger, however often imported."""
    logger = logging.getLogger("httpx")
    if not any(isinstance(f, RedactHttpxQuery) for f in logger.filters):
        logger.addFilter(RedactHttpxQuery())


install()

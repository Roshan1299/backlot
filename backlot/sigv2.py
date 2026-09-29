"""AWS Signature Version 2 verification for the S3 router — standard library only.

Real S3 still verifies it, in the header (``Authorization: AWS <key>:<signature>``) and in the query
(``AWSAccessKeyId``, ``Expires`` and ``Signature``): botocore's ``HmacV1Auth`` and
``HmacV1QueryAuth`` were served a listing, a bucket's ``?versioning``, an object and ListBuckets on
a bucket created that day in us-east-1, and a listing of the public bucket ``noaa-ghcn-pds``
(measured 2026-09-29). The signature is base64 HMAC-SHA1 over a string of the method,
``Content-MD5``, ``Content-Type``, the date, the ``x-amz-*`` headers and the resource, each spelled
here the way real spells them in the ``StringToSign`` it returns on a mismatch.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from datetime import datetime, timezone
from urllib.parse import parse_qsl

# The query parameters real names in the resource it signs. Measured 2026-09-29: every query
# parameter botocore's S3 model names, 72 of them, and `x-id`, `ACL`, `Versioning`, `x-amz-foo` and
# `X-Amz-Foo`, each sent as `?<name>=v` at a bucket's path and at a key's over a bad secret, and
# read out of the `StringToSign` real answered with; `website`, which refuses a value before
# anything is signed, was sent without one. These are the ones it named; the rest it left out, the
# listing's and ListMultipartUploads' own parameters among them, and names are matched with their
# case, so `ACL` and `Versioning` are left out too.
SUBRESOURCES = frozenset(
    {
        "abac",
        "accelerate",
        "acl",
        "analytics",
        "annotation",
        "annotationName",
        "attributes",
        "cors",
        "delete",
        "encryption",
        "intelligent-tiering",
        "inventory",
        "legal-hold",
        "lifecycle",
        "location",
        "logging",
        "metadataAnnotationTable",
        "metadataConfiguration",
        "metadataInventoryTable",
        "metadataJournalTable",
        "metadataTable",
        "metrics",
        "notification",
        "object-lock",
        "ownershipControls",
        "partNumber",
        "policy",
        "policyStatus",
        "publicAccessBlock",
        "replication",
        "requestPayment",
        "response-cache-control",
        "response-content-disposition",
        "response-content-encoding",
        "response-content-language",
        "response-content-type",
        "response-expires",
        "restore",
        "retention",
        "select",
        "select-type",
        "tagging",
        "torrent",
        "uploadId",
        "uploads",
        "versionId",
        "versioning",
        "versions",
        "website",
    }
)
# The ones real names with their value, decoded (`?response-content-type=a%2Fb` is signed as
# `response-content-type=a/b`); it names every other one alone whatever value was sent, `?acl=v`
# and `?acl=` both as `acl`, and a name sent twice once (same measurement).
VALUED = frozenset(
    {
        "partNumber",
        "response-cache-control",
        "response-content-disposition",
        "response-content-encoding",
        "response-content-language",
        "response-content-type",
        "response-expires",
        "select-type",
        "uploadId",
        "versionId",
    }
)
# The date forms real reads in `Date` and `x-amz-date`: RFC 1123, RFC 850 and the ISO 8601 basic
# form, each taken as a date and signed as sent (2026-09-29); anything else is no date at all.
_DATE_FORMATS = ("%a, %d %b %Y %H:%M:%S GMT", "%A, %d-%b-%y %H:%M:%S GMT", "%Y%m%dT%H%M%SZ")


def parse_date(value: str) -> datetime | None:
    """A ``Date`` or ``x-amz-date`` value as a UTC time, or ``None`` where real reads no date."""
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _resource(path: str, query: str) -> str:
    """The wire path, then the signed parameters sorted by name (see ``SUBRESOURCES``)."""
    named: dict[str, str] = {}
    for name, value in parse_qsl(query, keep_blank_values=True):
        if name in SUBRESOURCES:
            named.setdefault(name, value)
    parts = [f"{n}={named[n]}" if n in VALUED else n for n in sorted(named)]
    return path + ("?" + "&".join(parts) if parts else "")


def _amz_headers(headers: dict[str, str], query: str) -> str:
    """The ``x-amz-*`` headers and query parameters, lower-cased and sorted, one ``name:value`` a
    line. A query parameter real signs as a header too, and over a header of the same name: a GET
    carrying `x-amz-foo: 2` and `?x-amz-foo=1` was signed as `x-amz-foo:1` (2026-09-29), and a
    query-signed one's `X-Amz-Signature` as `x-amz-signature`."""
    named = {k: v.strip() for k, v in headers.items() if k.startswith("x-amz-")}
    for name, value in parse_qsl(query, keep_blank_values=True):
        if name.lower().startswith("x-amz-"):
            named[name.lower()] = value
    return "".join(f"{name}:{named[name]}\n" for name in sorted(named))


def string_to_sign(method: str, headers: dict[str, str], date: str, path: str, query: str) -> str:
    """What real signs: ``headers`` lower-cased, ``date`` the line it signs for the date (the
    ``Date`` header, empty beside an ``x-amz-date``, or a query's ``Expires``)."""
    return (
        f"{method}\n{headers.get('content-md5', '')}\n{headers.get('content-type', '')}\n{date}\n"
        + _amz_headers(headers, query)
        + _resource(path, query)
    )


def sign(secret: str, to_sign: str) -> str:
    return base64.b64encode(
        hmac.new(secret.encode("utf-8"), to_sign.encode("utf-8"), hashlib.sha1).digest()
    ).decode()

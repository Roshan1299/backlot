"""Auth helpers shared by the vendor routers.

Each vendor carries credentials differently (Slack bearer/query token, Google/GitHub
bearer, Atlassian Basic email:api_token, Linear a scheme-less API key). These helpers
extract the raw token, resolve it to a :class:`~backlot.acl.Caller` via the app's ACL, and
compute the caller's visible principal set. Error *shaping* (Slack's ``ok:false`` vs a
real 401) stays in the routers.
"""

from __future__ import annotations

import base64
import hmac
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request

from backlot import sigv2, sigv4
from backlot.acl import ANONYMOUS, Acl, Caller


def conn(request: Request) -> sqlite3.Connection:
    return request.app.state.conn


def acl(request: Request) -> Acl:
    return request.app.state.acl


def _authorization(request: Request) -> str | None:
    return request.headers.get("authorization")


def bearer_token(request: Request) -> str | None:
    """Parse ``Authorization: Bearer <t>`` or GitHub's legacy ``token <t>``."""
    hdr = _authorization(request)
    if not hdr:
        return None
    parts = hdr.split(None, 1)
    if len(parts) == 2 and parts[0].lower() in ("bearer", "token"):
        return parts[1].strip()
    return None


def api_key_token(request: Request) -> str | None:
    """Parse ``Authorization: <key>`` — with or without a ``Bearer`` prefix.

    Linear's GraphQL API carries a personal API key as the bare header value
    (``Authorization: lin_api_...``, no scheme) and an OAuth access token as
    ``Bearer <token>``, accepting both on the same header, so this accepts both too.
    Anything that is not a ``Bearer`` prefix is returned verbatim rather than having its
    first word stripped: to the real API the whole header value *is* the key, so a stray
    scheme fails to resolve instead of being quietly discarded.
    """
    hdr = (_authorization(request) or "").strip()
    if not hdr:
        return None
    parts = hdr.split(None, 1)
    if parts[0].lower() == "bearer":
        return parts[1].strip() or None if len(parts) == 2 else None
    return hdr


# How much of a Basic credential a request carries.
BASIC_ABSENT = "absent"  # no header, another scheme, or `Basic` with nothing to decode
BASIC_UNPARSEABLE = "unparseable"  # a value that is not one user and one password
BASIC_PAIR = "pair"  # exactly one non-empty user and one non-empty password


def _basic_value(request: Request) -> str | None:
    """The raw base64 payload of an ``Authorization: Basic`` header, or None for any other."""
    hdr = _authorization(request)
    if not hdr:
        return None
    parts = hdr.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "basic":
        return parts[1]
    return None


def _decoded_basic(request: Request) -> str | None:
    """The decoded ``user:pass`` of a Basic header, ``None`` when there is none to decode and
    ``""`` for a payload that is not base64 — a value that was there and could not be read."""
    value = _basic_value(request)
    if not value:
        return None
    try:
        return base64.b64decode(value).decode("utf-8", "replace")
    except (ValueError, UnicodeDecodeError):
        return ""


def basic_password(request: Request) -> tuple[str | None, str | None]:
    """Parse ``Authorization: Basic base64(user:pass)`` -> (user, pass)."""
    decoded = _decoded_basic(request)
    if not decoded:
        return None, None
    user, _, pw = decoded.partition(":")
    return user, pw


def basic_credential_kind(request: Request) -> str:
    """Which of :data:`BASIC_ABSENT` / :data:`BASIC_UNPARSEABLE` / :data:`BASIC_PAIR` the request
    carries.

    Confluence answers the three differently — a pair it read and rejected is its 403, a value it
    could not read is a 401, and no credential at all is the 403 again — so which one a request
    holds decides which refusal it draws. Measured against ecosystem.atlassian.net and
    brekkylab.atlassian.net on 2026-09-04, which is where the split is observable at all: a single
    colon with both halves non-empty is the pair; an empty user, an empty password, no colon, a
    second colon, or a payload that is not base64 is unparseable; and a missing header, an unknown
    scheme and a `Basic` with nothing after it are no credential.
    """
    if not _basic_value(request):
        return BASIC_ABSENT
    decoded = _decoded_basic(request)
    if not decoded:  # a payload that is there and does not decode
        return BASIC_UNPARSEABLE
    user, _, pw = decoded.partition(":")
    if user and pw and ":" not in pw:
        return BASIC_PAIR
    return BASIC_UNPARSEABLE


def basic_names_a_user(request: Request) -> bool:
    """Whether the request presented a Basic credential naming somebody.

    Jira reports one it could not resolve in ``X-Seraph-LoginReason``, and the header is keyed on
    the username alone: a value carrying a non-empty user before its first colon draws it whatever
    follows, and one with an empty user, or with no colon at all, draws nothing. Measured on both
    sites on 2026-09-04.
    """
    decoded = _decoded_basic(request) or ""
    user, colon, _ = decoded.partition(":")
    return bool(user and colon)


def slack_bearer_token(request: Request) -> str | None:
    """Parse the ``Authorization`` header the way Slack does, which is stricter than
    :func:`bearer_token`: the scheme must be exactly ``Bearer``, separated from the token by one or
    more SPACES.

    Measured against slack.com (a bogus token is enough — the presented/absent split needs no
    account). ``invalid_auth`` means the header counted as a credential, ``not_authed`` means it
    did not::

        Bearer <t>      invalid_auth      bearer <t>      not_authed
        Bearer  <t>     invalid_auth      BEARER <t>      not_authed
        ' Bearer <t>'   invalid_auth      token <t>       not_authed
        Bearer <t>' '   invalid_auth      Bearer<TAB><t>  not_authed
                                          Bearer<t>       not_authed

    A tab is not a space to Slack, so the generic whitespace split in :func:`bearer_token` is
    wrong here, and so is its case-insensitive scheme match. That function stays permissive because
    GitHub really does accept ``token <t>`` and RFC 7235 really does make the scheme
    case-insensitive; Slack implements neither. Of the five spellings above that Slack refuses,
    sharing it would authenticate four — every one but ``Bearer<t>``, which it does not take
    either — so a client sending ``Authorization: token <xoxb>`` would pass every test here and
    reach nothing in production.
    """
    hdr = (_authorization(request) or "").strip()
    if not hdr.startswith("Bearer "):
        return None
    return hdr[len("Bearer ") :].strip() or None


def slack_token(request: Request) -> str | None:
    """Slack accepts the token as a bearer header, query param, or form field. The official
    slack-go SDK (and Slack's own clients) post it as the ``token`` form field, so fall back to
    the form stashed on ``request.state._form`` by the slack-form middleware.

    Both names are case-sensitive too: ``?TOKEN=`` and a ``TOKEN`` form field are ``not_authed``
    live, which the exact-key lookups below already answer."""
    form = getattr(request.state, "_form", None)
    form_field = form.get("token") if form else None
    return slack_bearer_token(request) or request.query_params.get("token") or form_field


def resolve_bearer(request: Request) -> Caller | None:
    return acl(request).resolve(bearer_token(request))


def require_bearer(request: Request, detail: str) -> Caller:
    """Resolve a bearer token or raise 401 with the VENDOR's own message.

    ``detail`` is a parameter rather than something this function picks, because the message is
    part of the emulated surface: GitHub says "Bad credentials", Google "Invalid Credentials",
    Atlassian "Unauthorized", and a client that string-matches its vendor's error has to keep
    matching. Each router states its own once (see ``tests/test_endpoints.py``).
    """
    caller = resolve_bearer(request)
    if caller is None:
        raise HTTPException(status_code=401, detail=detail)
    return caller


def atlassian_bearer_token(request: Request) -> str | None:
    """Parse the ``Authorization`` header the way a ``<site>.atlassian.net`` gateway does, which
    is stricter than :func:`bearer_token` and strict differently from :func:`slack_bearer_token`.

    Measured against ecosystem.atlassian.net and brekkylab.atlassian.net on 2026-09-04 (a bogus
    token is enough — a recognised credential is refused with a 403 and an unrecognised one is
    served anonymously, so the answer says which the site read)::

        Bearer <t>      read            bearer <t>      not read
        ' Bearer <t>'   read            BEARER <t>      not read
        Bearer <t>' '   read            token <t>       not read
                                        OAuth <t>       not read
                                        Bearer  <t>     not read
                                        Bearer<TAB><t>  not read
                                        Bearer<t>       not read
                                        Bearer          not read

    The scheme is case-sensitive and separated from the token by exactly one space. Sharing
    :func:`bearer_token` would authenticate five of those spellings — every "not read" row above
    but ``OAuth <t>``, ``Bearer<t>`` and the bare ``Bearer`` — so a client sending
    ``Authorization: token <t>`` would pass every test here and read nothing in production. Slack
    refuses a different set: it takes the double space this refuses.

    The leading ``strip`` is what serves the ``Bearer <t>' '`` row, so the token needs no second
    one: nothing with trailing whitespace survives to reach it.
    """
    hdr = (_authorization(request) or "").strip()
    if not hdr.startswith("Bearer "):
        return None
    rest = hdr[len("Bearer ") :]
    if not rest or rest[0].isspace():
        return None
    return rest


def atlassian_bearer_unreadable(request: Request) -> bool:
    """Whether the request carries a bearer the site would read and Backlot cannot resolve.

    A Backlot token is an opaque string with no dots, and that is the shape the gateway reports as
    unreadable — measured with ``usr-<hex>`` itself. A token shaped like a complete signed JWS is
    read and then rejected with a 401 instead; Backlot issues none, and reproducing Atlassian
    Connect's accept boundary would mean inventing the space between the shapes measured, so a
    JWT-shaped bearer takes the 403 here too.
    """
    token = atlassian_bearer_token(request)
    return bool(token) and acl(request).resolve(token) is None


def atlassian_caller(request: Request) -> Caller:
    """The caller for an Atlassian read: Basic ``email:api_token`` or
    :func:`atlassian_bearer_token`, and :data:`backlot.acl.ANONYMOUS` when neither resolves.

    The bearer is NOT an OAuth 3LO token standing in for Atlassian's own. A 3LO token goes to
    ``api.atlassian.com/ex/jira/{cloudid}/…``, which Backlot does not serve; on the
    ``<site>.atlassian.net`` surface it does serve, a bearer is read as a Connect session JWT, and
    an opaque Backlot token is one the gateway cannot read at all — which is the ``403``
    :func:`atlassian_bearer_unreadable` reports.

    No refusal here, unlike :func:`require_bearer`: the two Atlassian APIs disagree about what an
    unresolved credential means. Jira drops the caller to anonymous and answers the request; only
    Confluence refuses. Each router decides for itself, so this reports the identity and nothing
    else.
    """
    bearer = atlassian_bearer_token(request)
    return resolve_basic(request) or acl(request).resolve(bearer) or ANONYMOUS


def resolve_api_key(request: Request) -> Caller | None:
    return acl(request).resolve(api_key_token(request))


def resolve_basic(request: Request) -> Caller | None:
    """Atlassian: the api_token is the password, and the username has to be its own account.

    Both halves, because that is what the real service requires. Measured against a real Atlassian
    Cloud site (``GET /rest/api/3/myself``) with a user API token: ``email:token`` answers 200,
    while an empty password, a wrong password, and a valid token under someone else's address all
    answer 401. Matched case-insensitively, which is also measured: the same token under
    ``AVA.CHEN@…`` and ``Ava.chen@…`` answers 200 as that same account.

    The admin/service token is the one caller with no address — ``Acl.resolve`` gives it
    ``Caller(email=None)`` — so there is nothing to match a username against and any username is
    taken. It has no vendor analogue to be faithful to: it is Backlot's own full-crawl identity,
    and it is what lets an Atlassian client send the placeholder username its config demands.
    """
    user, pw = basic_password(request)
    caller = acl(request).resolve(pw)
    if caller is None:
        return None
    if caller.email is None:
        return caller
    return caller if (user or "").casefold() == caller.email.casefold() else None


def visible_ids(request: Request, caller: Caller) -> set[str] | None:
    return acl(request).visible_ids(conn(request), caller)


@dataclass(frozen=True)
class SigV4Refusal:
    """A credential real S3 refuses: its code, the message it sends and the members after it."""

    code: str
    message: str
    members: tuple[tuple[str, str], ...] = ()


_HEADER_MALFORMED = "The authorization header is malformed; "
_QUERY_CREDENTIAL = "Error parsing the X-Amz-Credential parameter; "
_CREDENTIAL_FORMAT = (
    'the Credential is mal-formed; expecting "<YOUR-AKID>/YYYYMMDD/REGION/SERVICE/aws4_request".'
)
_QUERY_PARAMETERS = (
    "X-Amz-Credential",
    "X-Amz-Signature",
    "X-Amz-Date",
    "X-Amz-SignedHeaders",
    "X-Amz-Expires",
)
_SERVER_TIME = "%Y-%m-%dT%H:%M:%SZ"
# The longest `X-Amz-Expires` real takes: 604800 was served and 604801 refused (2026-09-29).
_A_WEEK = 604800
# How far ahead a presign's date may be: real served one five minutes ahead and refused one an hour
# ahead (2026-09-29); the header's own skew window stands in for the boundary between the two.
_NOT_YET_VALID = 900


def _credential_fault(credential: str) -> tuple[str, tuple[tuple[str, str], ...]] | None:
    """What real names wrong with a credential scope, and the members it names it with, or ``None``.

    In real's order: the five parts, the region, the service, the terminal. The region is the one
    this server presents, and real answers any other the way it answered `us-west-2`, `eu-west-1`,
    `US-EAST-1`, `zz` and `us-east-1a` on `s3.us-east-1.amazonaws.com`, naming the region sent and
    the one expected, the latter again as `Region`, and an empty one with a message of its own and
    no member; a region beside a service, a terminal, a scope date or an access key that is wrong
    is refused for the region (measured 2026-09-29, in the header and in the query alike)."""
    bits = credential.split("/")
    if len(bits) != 5:
        return _CREDENTIAL_FORMAT, ()
    if not bits[2]:
        return "a non-empty region must be provided in the credential.", ()
    if bits[2] != sigv4.REGION:
        message = f"the region '{bits[2]}' is wrong; expecting '{sigv4.REGION}'"
        return message, (("Region", sigv4.REGION),)
    if bits[3] != "s3":
        return f'incorrect service "{bits[3]}". This endpoint belongs to "s3".', ()
    if bits[4] != "aws4_request":
        return f'incorrect terminal "{bits[4]}". This endpoint uses "aws4_request".', ()
    return None


def _hex_bytes(text: str) -> str:
    return " ".join(f"{b:02x}" for b in text.encode("utf-8"))


_ONE_MECHANISM = (
    "Only one auth mechanism allowed; only the X-Amz-Algorithm query parameter, "
    "Signature query string parameter or the Authorization header should be specified"
)
_NO_SPACE = "Authorization header is invalid -- one and only one ' ' (space) required"
_NO_DATE = "AWS authentication requires a valid Date or x-amz-date header"
_NO_KEY = "The AWS Access Key Id you provided does not exist in our records."
_MISMATCH = (
    "The request signature we calculated does not match the signature you provided. "
    "Check your key and signing method."
)
# A Signature Version 2 query's `Expires` is read as seconds since the epoch in an int32: 2147483647
# was a date and 2147483648 not, nor `1e9` or ` 1`, where `01790000000` and `-1` were (2026-09-29).
_INT32 = 2**31


def signed_with_v2(request: Request) -> bool:
    """Whether the credential a request carries is a Signature Version 2 one, header or query."""
    scheme = request.headers.get("authorization", "").partition(" ")[0]
    if scheme:
        return scheme.upper() == "AWS"
    return "Signature" in request.query_params and "X-Amz-Algorithm" not in request.query_params


def _wire(request: Request) -> tuple[str, str]:
    """The path and query as uvicorn received them. Both halves of what is signed come off the
    wire, not off `request.url`: Starlette rebuilds that URL from the DECODED path, so a key
    containing `%3F` turns into a `?` that splits it — `/q%3Fx.txt` reads back as path `/q` with
    query `x.txt`, a query the client never signed."""
    raw = request.scope.get("raw_path")
    path = raw.decode("ascii") if raw else request.url.path
    return path, request.scope.get("query_string", b"").decode("ascii")


def _skewed(request_time: str, now: datetime) -> SigV4Refusal:
    return SigV4Refusal(
        "RequestTimeTooSkewed",
        "The difference between the request time and the current time is too large.",
        (
            ("RequestTime", request_time),
            ("ServerTime", now.strftime(_SERVER_TIME)),
            ("MaxAllowedSkewMilliseconds", "900000"),
        ),
    )


def resolve_sigv4(request: Request) -> tuple[Caller | None, SigV4Refusal | None]:
    """Verify an S3 request's credential: Signature Version 4 in the header or the query, or
    Signature Version 2 in either (``_resolve_v2_header``, ``_resolve_v2_query``).

    Returns ``(caller, None)`` on a valid signature, ``(ANONYMOUS, None)`` for a request carrying
    no credential at all, and ``(None, refusal)`` otherwise. Real S3 reads an unsigned request as
    an anonymous caller's rather than refusing it, and so does this: what an anonymous caller can
    see is decided where every caller's is, and it can see nothing. A V4 presigned request is one
    carrying `X-Amz-Algorithm` and a V2 one a request carrying `Signature`; an `X-Amz-Signature`
    without the first and an `AWSAccessKeyId` without the second are unsigned on real (a public
    bucket's listing answered the one and a bucket's own owner was refused the other as anonymous).

    Each refusal is real's own code, message and members, and they come in real's order:
    measured 2026-09-29 against `s3.us-east-1.amazonaws.com`, one fault at a time and, where two
    could meet, the two together. A header beside `X-Amz-Algorithm` or `Signature` in the query is
    refused first, whatever either says, and the two query forms beside each other. For a header it
    is then its one space and its scheme, matched without case (anything but `AWS4-HMAC-SHA256` and
    V2's `AWS` is `InvalidArgument`, "Unsupported Authorization Type", Bearer and `AWS4-HMAC-SHA512`
    alike), a missing or unreadable `x-amz-date` (an `AccessDenied`), the clock skew, its three
    components, the credential scope's shape, region, service and terminal (``_credential_fault``),
    a scope date that is not the request's, the access key and the signature: the date and the
    skew come ahead of every fault in the header's body. For a query it is the algorithm, the six
    parameters, the date, `X-Amz-Expires` as a number, as not negative and as a week at most, a
    date not yet valid, the expiry, the credential scope's shape, region, service and terminal, its
    date, the access key and the signature. A signature mismatch names the string this server
    signed and the canonical request it signed it over, as bytes too, the way real names its own.
    The canonical URI and query are the raw wire path and query string (S3 signs the path
    verbatim, see ``_wire``)."""
    hdrs = {k.lower(): v for k, v in request.headers.items()}
    qs = request.query_params
    now = datetime.now(timezone.utc)
    authz = hdrs.get("authorization", "")
    presigned = "X-Amz-Algorithm" in qs
    v2_query = "Signature" in qs
    if authz:
        argument = (("ArgumentName", "Authorization"), ("ArgumentValue", authz))
        if presigned or v2_query:
            return None, SigV4Refusal("InvalidArgument", _ONE_MECHANISM, argument)
        scheme, space, rest = authz.partition(" ")
        if not space:
            return None, SigV4Refusal("InvalidArgument", _NO_SPACE, argument)
        if scheme.upper() == "AWS":
            return _resolve_v2_header(request, hdrs, rest, argument, now)
        if scheme.upper() != sigv4.ALGORITHM:
            return None, SigV4Refusal("InvalidArgument", "Unsupported Authorization Type", argument)
        algorithm = scheme
        amz_date = hdrs.get("x-amz-date", "")
        request_time = sigv4.parse_amz_date(amz_date)
        if request_time is None:
            return None, SigV4Refusal("AccessDenied", _NO_DATE)
        if sigv4.is_skewed(request_time, now):
            return None, _skewed(amz_date, now)
        parsed = sigv4.parse_authorization(authz)
        if not parsed:
            message = (
                "the authorization header requires three components: Credential, SignedHeaders, "
                "and Signature."
            )
            return None, SigV4Refusal("AuthorizationHeaderMalformed", _HEADER_MALFORMED + message)
        credential = parsed["credential"]
        fault = _credential_fault(credential)
        if fault:
            message, members = fault
            return None, SigV4Refusal(
                "AuthorizationHeaderMalformed", _HEADER_MALFORMED + message, members
            )
        if credential.split("/")[1] != amz_date[:8]:
            message = "Invalid credential date. Date is not the same as X-Amz-Date."
            return None, SigV4Refusal("AuthorizationHeaderMalformed", _HEADER_MALFORMED + message)
        signed_headers, signature = parsed["signed_headers"], parsed["signature"]
        payload_hash = hdrs.get("x-amz-content-sha256", "UNSIGNED-PAYLOAD")
    elif presigned:
        if v2_query:
            # Named without an `ArgumentValue`, there being no header to name (same date).
            return None, SigV4Refusal(
                "InvalidArgument", _ONE_MECHANISM, (("ArgumentName", "Authorization"),)
            )
        algorithm = sigv4.ALGORITHM
        if qs.get("X-Amz-Algorithm") != sigv4.ALGORITHM:
            message = 'X-Amz-Algorithm only supports "AWS4-HMAC-SHA256 and AWS4-ECDSA-P256-SHA256"'
            return None, SigV4Refusal("AuthorizationQueryParametersError", message)
        if any(name not in qs for name in _QUERY_PARAMETERS):
            message = (
                "Query-string authentication version 4 requires the X-Amz-Algorithm, "
                "X-Amz-Credential, X-Amz-Signature, X-Amz-Date, X-Amz-SignedHeaders, and "
                "X-Amz-Expires parameters."
            )
            return None, SigV4Refusal("AuthorizationQueryParametersError", message)
        amz_date = qs["X-Amz-Date"]
        request_time = sigv4.parse_amz_date(amz_date)
        if request_time is None:
            message = "X-Amz-Date must be in the ISO8601 Long Format \"yyyyMMdd'T'HHmmss'Z'\""
            return None, SigV4Refusal("AuthorizationQueryParametersError", message)
        try:
            expires_in = int(qs["X-Amz-Expires"])
        except ValueError:
            message = "X-Amz-Expires should be a number"
            return None, SigV4Refusal("AuthorizationQueryParametersError", message)
        if expires_in < 0:
            message = "X-Amz-Expires must be non-negative"
            return None, SigV4Refusal("AuthorizationQueryParametersError", message)
        if expires_in > _A_WEEK:
            message = (
                "X-Amz-Expires must be less than a week (in seconds); that is, the given "
                "X-Amz-Expires must be less than 604800 seconds"
            )
            return None, SigV4Refusal("AuthorizationQueryParametersError", message)
        expires_at = (request_time + timedelta(seconds=expires_in)).strftime(_SERVER_TIME)
        if (request_time - now).total_seconds() > _NOT_YET_VALID:
            return None, SigV4Refusal(
                "AccessDenied",
                "Request is not yet valid",
                (
                    ("X-Amz-Date", str(int(request_time.timestamp() * 1000))),
                    ("Expires", expires_at),
                    ("ServerTime", now.strftime(_SERVER_TIME)),
                ),
            )
        if (now - request_time).total_seconds() > expires_in:
            return None, SigV4Refusal(
                "AccessDenied",
                "Request has expired",
                (
                    ("X-Amz-Expires", qs["X-Amz-Expires"]),
                    ("Expires", expires_at),
                    ("ServerTime", now.strftime(_SERVER_TIME)),
                ),
            )
        credential = qs["X-Amz-Credential"]
        fault = _credential_fault(credential)
        if fault:
            message, members = fault
            return None, SigV4Refusal(
                "AuthorizationQueryParametersError", _QUERY_CREDENTIAL + message, members
            )
        if credential.split("/")[1] != amz_date[:8]:
            message = (
                f'Invalid credential date "{credential.split("/")[1]}". This date is not the same '
                f'as X-Amz-Date: "{amz_date[:8]}".'
            )
            return None, SigV4Refusal("AuthorizationQueryParametersError", message)
        signed_headers = qs["X-Amz-SignedHeaders"]
        signature = qs["X-Amz-Signature"]
        payload_hash = "UNSIGNED-PAYLOAD"
    elif v2_query:
        return _resolve_v2_query(request, hdrs, now)
    else:
        return ANONYMOUS, None
    access_key, date_stamp, region = credential.split("/")[:3]
    resolved = acl(request).resolve_access_key(access_key)
    if resolved is None:
        return None, SigV4Refusal("InvalidAccessKeyId", _NO_KEY, (("AWSAccessKeyId", access_key),))
    caller, secret = resolved
    path, query = _wire(request)
    canonical = sigv4.canonical_request(
        request.method, path, query, hdrs, signed_headers, payload_hash
    )
    to_sign = sigv4.string_to_sign(amz_date, date_stamp, region, canonical, algorithm)
    if not hmac.compare_digest(
        sigv4.sign(secret, date_stamp, region, to_sign).encode(), signature.encode()
    ):
        return None, SigV4Refusal(
            "SignatureDoesNotMatch",
            _MISMATCH,
            (
                ("AWSAccessKeyId", access_key),
                ("StringToSign", to_sign),
                ("SignatureProvided", signature),
                ("StringToSignBytes", _hex_bytes(to_sign)),
                ("CanonicalRequest", canonical),
                ("CanonicalRequestBytes", _hex_bytes(canonical)),
            ),
        )
    return caller, None


def _resolve_v2_header(
    request: Request, hdrs: dict[str, str], rest: str, argument: tuple, now: datetime
) -> tuple[Caller | None, SigV4Refusal | None]:
    """`Authorization: AWS <key>:<signature>`, refused as real refuses it (measured 2026-09-29).

    In real's order: a second space after the scheme, which V4's header may carry and this one may
    not; a body that is not one key, one colon and a signature (`garbage`, `<key>:` and `<key>:a:b`
    alike, where an empty key reads as a key); the date; the skew, naming the date as sent; the
    access key; the signature. The date is `x-amz-date` when one is sent, readable or not, and
    otherwise `Date`, in any of the forms ``sigv2.parse_date`` reads, and the line signed for it is
    empty beside an `x-amz-date`, which is signed among the `x-amz-*` headers instead."""
    if rest.startswith(" "):
        return None, SigV4Refusal("InvalidArgument", _NO_SPACE, argument)
    access_key, colon, signature = rest.partition(":")
    if not colon or not signature or ":" in signature:
        message = "AWS authorization header is invalid.  Expected AwsAccessKeyId:signature"
        return None, SigV4Refusal("InvalidArgument", message, argument)
    if "x-amz-date" in hdrs:
        date, date_line = hdrs["x-amz-date"], ""
    else:
        date = date_line = hdrs.get("date", "")
    request_time = sigv2.parse_date(date)
    if request_time is None:
        return None, SigV4Refusal("AccessDenied", _NO_DATE)
    if sigv4.is_skewed(request_time, now):
        return None, _skewed(date, now)
    return _verify_v2(request, hdrs, access_key, signature, date_line)


def _resolve_v2_query(
    request: Request, hdrs: dict[str, str], now: datetime
) -> tuple[Caller | None, SigV4Refusal | None]:
    """`?AWSAccessKeyId=…&Expires=…&Signature=…`, refused as real refuses it (measured 2026-09-29).

    In real's order: `Expires` or `AWSAccessKeyId` missing beside the `Signature`; an `Expires`
    that is not a date (``_INT32``), naming it as sent; the expiry, naming it as a time; the access
    key; the signature, over a string whose date line is `Expires` as sent. A query has no clock
    skew: one ten years ahead was served."""
    qs = request.query_params
    if "Expires" not in qs or "AWSAccessKeyId" not in qs:
        message = (
            "Query-string authentication requires the Signature, Expires and AWSAccessKeyId "
            "parameters"
        )
        return None, SigV4Refusal("AccessDenied", message)
    raw = qs["Expires"]
    # Leading zeros come off before the value is read, as they do for `max-keys`, so a run of them
    # never reaches `int()`, which refuses past 4300 digits.
    digits = raw.removeprefix("-").lstrip("0") or "0"
    readable = re.fullmatch(r"-?[0-9]+", raw) is not None and len(digits) <= 10
    expires = int(digits) * (-1 if raw.startswith("-") else 1) if readable else _INT32
    if not -_INT32 <= expires < _INT32:
        message = f"Invalid date (should be seconds since epoch): {raw}"
        return None, SigV4Refusal("AccessDenied", message)
    if now.timestamp() > expires:
        return None, SigV4Refusal(
            "AccessDenied",
            "Request has expired",
            (
                ("Expires", datetime.fromtimestamp(expires, timezone.utc).strftime(_SERVER_TIME)),
                ("ServerTime", now.strftime(_SERVER_TIME)),
            ),
        )
    return _verify_v2(request, hdrs, qs["AWSAccessKeyId"], qs["Signature"], raw)


# Where `backlot.routers.s3` is mounted, which real S3 has no counterpart of.
_S3_MOUNT = "/s3"


def _verify_v2(
    request: Request, hdrs: dict[str, str], access_key: str, signature: str, date_line: str
) -> tuple[Caller | None, SigV4Refusal | None]:
    """The access key and then the signature, the members named as real names them for V2: the
    string signed and its bytes, and no canonical request, V2 having none.

    Real signs the path it was sent, which on real is the bucket and the key. Here that path sits
    under ``_S3_MOUNT``, and a client signs either what is under it or all of it: boto3 signs the
    bucket and the key it addresses (its `auth_path`) and for ListBuckets the URL's own path, and a
    signer handed a URL signs that URL's path. So a signature over either verifies, and a mismatch
    names the one real would sign, the path under the mount. boto3's path-style signature for a
    bucket's own operations carries a slash after the bucket that its URL does not, and real
    refused that as a mismatch (2026-09-29), which this is too."""
    resolved = acl(request).resolve_access_key(access_key)
    if resolved is None:
        return None, SigV4Refusal("InvalidAccessKeyId", _NO_KEY, (("AWSAccessKeyId", access_key),))
    caller, secret = resolved
    path, query = _wire(request)
    under = path[len(_S3_MOUNT) :] if path.startswith(_S3_MOUNT) else path
    signed = [
        sigv2.string_to_sign(request.method, hdrs, date_line, candidate, query)
        for candidate in (under or "/", path)
    ]
    to_sign = signed[0]
    if not any(
        hmac.compare_digest(sigv2.sign(secret, candidate).encode(), signature.encode())
        for candidate in signed
    ):
        return None, SigV4Refusal(
            "SignatureDoesNotMatch",
            _MISMATCH,
            (
                ("AWSAccessKeyId", access_key),
                ("StringToSign", to_sign),
                ("SignatureProvided", signature),
                ("StringToSignBytes", _hex_bytes(to_sign)),
            ),
        )
    return caller, None

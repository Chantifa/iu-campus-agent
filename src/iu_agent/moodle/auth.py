"""Obtain and store a Moodle web-service token.

myCampus uses SSO (OAuth2 via auth.iu.org, ``typeoflogin = 3``), so the classic
``login/token.php`` username/password endpoint does not work. The Moodle mobile app solves this
with the *launch* flow, which this module reproduces:

1. open ``<site>/admin/tool/mobile/launch.php`` with the query
   ``service=moodle_mobile_app&passport=<random>&urlscheme=moodlemobile``
   in a normal browser and log in through the IU SSO,
2. Moodle redirects to ``moodlemobile://token=<base64>`` where the base64 payload is
   ``md5(site + passport):::<token>:::<privatetoken>``,
3. paste that URL (or just the token) into ``iu-agent moodle login`` which validates and stores it.

Alternatively a token can be created manually under *Preferences -> Security keys* in myCampus.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import re
import stat
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iu_agent.config import Settings

_TOKEN_RE = re.compile(r"^[a-f0-9]{32}$")


@dataclass
class MoodleToken:
    token: str
    private_token: str | None = None
    site_hash_valid: bool | None = None


def make_passport() -> str:
    """Random passport like the official app (``Math.random() * 1000``)."""
    return str(random.random() * 1000)


def launch_url(
    base_url: str, passport: str, service: str = "moodle_mobile_app", launch_path: str | None = None
) -> str:
    url = launch_path or f"{base_url.rstrip('/')}/admin/tool/mobile/launch.php"
    query = urllib.parse.urlencode({"service": service, "passport": passport, "urlscheme": "moodlemobile"})
    return f"{url}?{query}"


def parse_launch_response(text: str, base_url: str | None = None, passport: str | None = None) -> MoodleToken:
    """Accept ``moodlemobile://token=...``, the bare base64 payload or a bare 32-char token."""
    text = text.strip().strip("\"'")
    match = re.search(r"token=([^&\s]+)", text)
    raw = urllib.parse.unquote(match.group(1) if match else text)
    if _TOKEN_RE.match(raw):
        return MoodleToken(raw)
    try:
        decoded = base64.b64decode(raw + "=" * (-len(raw) % 4)).decode("utf-8")
    except Exception as exc:  # binascii.Error, UnicodeDecodeError
        raise ValueError("That does not look like a Moodle launch URL, token payload or token.") from exc
    parts = decoded.split(":::")
    if len(parts) < 2 or not parts[1]:
        raise ValueError(
            "Decoded payload does not contain a token (expected sitehash:::token:::privatetoken)."
        )
    site_hash, token = parts[0], parts[1]
    private_token = parts[2] if len(parts) > 2 and parts[2] else None
    valid: bool | None = None
    if base_url and passport:
        expected = hashlib.md5((base_url.rstrip("/") + passport).encode("utf-8")).hexdigest()  # noqa: S324
        valid = expected == site_hash
    return MoodleToken(token=token, private_token=private_token, site_hash_valid=valid)


# ----------------------------------------------------------------------------- persistence
def load_token(settings: Settings) -> str | None:
    if settings.moodle_token:
        return settings.moodle_token.get_secret_value()
    info = token_info(settings)
    return info.get("token") if info else None


def token_info(settings: Settings) -> dict[str, Any] | None:
    path = settings.moodle_token_path
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_token(settings: Settings, token: MoodleToken, site_info: dict[str, Any] | None = None) -> Path:
    path = settings.moodle_token_path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "site": settings.moodle_url,
        "token": token.token,
        "private_token": token.private_token,
        "fullname": (site_info or {}).get("fullname"),
        "username": (site_info or {}).get("username"),
        "userid": (site_info or {}).get("userid"),
        "sitename": (site_info or {}).get("sitename"),
    }
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return path


def clear_token(settings: Settings) -> bool:
    path = settings.moodle_token_path
    if path.exists():
        path.unlink()
        return True
    return False

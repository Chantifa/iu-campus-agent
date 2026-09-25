"""Minimal client for the Moodle REST web-service API (``/webservice/rest/server.php``).

myCampus classic (https://mycampus-classic.iu.org) is a Moodle site with web services and the
mobile service enabled (see ``tool_mobile_get_public_config``). Every call needs a web-service
token, see :mod:`iu_agent.moodle.auth` for how to obtain one through the SSO login.
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class MoodleError(RuntimeError):
    def __init__(self, message: str, errorcode: str | None = None) -> None:
        super().__init__(message)
        self.errorcode = errorcode


def flatten_params(params: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Encode nested values the way Moodle expects them: ``courseids[0]=12``, ``options[0][name]=x``."""
    flat: dict[str, str] = {}
    for key, value in params.items():
        name = f"{prefix}[{key}]" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_params(value, name))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if isinstance(item, (dict, list, tuple)):
                    flat.update(flatten_params({index: item}, name))
                else:
                    flat[f"{name}[{index}]"] = _scalar(item)
        elif value is not None:
            flat[name] = _scalar(value)
    return flat


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def with_token(fileurl: str, token: str) -> str:
    separator = "&" if "?" in fileurl else "?"
    return f"{fileurl}{separator}token={token}"


class MoodleClient:
    def __init__(
        self, base_url: str, token: str, timeout: float = 120.0, http: httpx.Client | None = None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.http = http or httpx.Client(timeout=timeout, follow_redirects=True)

    # ------------------------------------------------------------------ public (no token) helpers
    @staticmethod
    def public_config(
        base_url: str, timeout: float = 30.0, http: httpx.Client | None = None
    ) -> dict[str, Any]:
        """``tool_mobile_get_public_config``: login type, launch URL, identity providers ..."""
        client = http or httpx.Client(timeout=timeout, follow_redirects=True)
        args = json.dumps([{"index": 0, "methodname": "tool_mobile_get_public_config", "args": {}}])
        response = client.get(f"{base_url.rstrip('/')}/lib/ajax/service-nologin.php", params={"args": args})
        response.raise_for_status()
        payload = response.json()
        entry = payload[0] if isinstance(payload, list) else payload
        if entry.get("error"):
            exception = entry.get("exception") or {}
            raise MoodleError(
                exception.get("message", "public config unavailable"), exception.get("errorcode")
            )
        return entry["data"]

    # ------------------------------------------------------------------ generic call
    def call(self, function: str, **params: Any) -> Any:
        data = {
            "wstoken": self.token,
            "wsfunction": function,
            "moodlewsrestformat": "json",
            **flatten_params(params),
        }
        response = self.http.post(f"{self.base_url}/webservice/rest/server.php", data=data)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and ("exception" in payload or "errorcode" in payload):
            raise MoodleError(
                payload.get("message") or payload.get("error") or "Moodle web service error",
                payload.get("errorcode"),
            )
        return payload

    # ------------------------------------------------------------------ typed helpers
    def site_info(self) -> dict[str, Any]:
        return self.call("core_webservice_get_site_info")

    def user_courses(self, userid: int) -> list[dict[str, Any]]:
        return self.call("core_enrol_get_users_courses", userid=userid)

    def course_contents(self, courseid: int) -> list[dict[str, Any]]:
        return self.call("core_course_get_contents", courseid=courseid)

    def pages(self, courseids: list[int]) -> list[dict[str, Any]]:
        return self.call("mod_page_get_pages_by_courses", courseids=courseids).get("pages", [])

    def assignments(self, courseids: list[int]) -> list[dict[str, Any]]:
        result = self.call("mod_assign_get_assignments", courseids=courseids)
        assignments: list[dict[str, Any]] = []
        for course in result.get("courses", []):
            assignments.extend(course.get("assignments", []))
        return assignments

    def forums(self, courseids: list[int]) -> list[dict[str, Any]]:
        return self.call("mod_forum_get_forums_by_courses", courseids=courseids)

    def discussions(self, forumid: int) -> list[dict[str, Any]]:
        return self.call("mod_forum_get_forum_discussions", forumid=forumid).get("discussions", [])

    def download(self, fileurl: str) -> bytes:
        """Download a ``pluginfile`` URL returned by the API (the token is passed as query parameter)."""
        response = self.http.get(with_token(fileurl, self.token))
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if content_type.startswith("application/json"):
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if isinstance(payload, dict) and ("exception" in payload or "errorcode" in payload):
                raise MoodleError(payload.get("message", "download failed"), payload.get("errorcode"))
        return response.content

    def close(self) -> None:
        self.http.close()

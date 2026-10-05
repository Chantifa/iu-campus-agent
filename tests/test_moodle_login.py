import base64
import hashlib
import urllib.parse

import httpx
import pytest

from iu_agent.config import Settings
from iu_agent.moodle.login import (
    InteractiveLoginRequired,
    LoginError,
    find_login_form,
    host_allowed,
    password_login,
)

MOODLE = "https://mycampus-classic.iu.org"
MOODLE_HOST = "mycampus-classic.iu.org"
TOKEN = "b" * 32
GOOD_PASSWORD = "correct horse"

NATIVE_FORM = (
    '<form action="{moodle}/login/index.php" method="post" id="login">'
    '<input type="hidden" name="anchor" value=""><input type="hidden" name="logintoken" value="lt1">'
    '<input type="text" name="username"><input type="password" name="password">'
    '<button type="submit" id="loginbtn">Log in</button></form>'
)


class FakeCampus:
    """A Moodle site plus an Auth0-style identity provider behind one mock transport."""

    def __init__(
        self,
        *,
        token_endpoint_ok: bool = False,
        native_ok: bool = False,
        idp_host: str = "auth.iu.org",
        mfa: bool = False,
        captcha: bool = False,
    ) -> None:
        self.token_endpoint_ok = token_endpoint_ok
        self.native_ok = native_ok
        self.idp_host = idp_host
        self.mfa = mfa
        self.captcha = captcha
        self.logged_in = False
        self.native_failed = False
        self.password_posts: list[tuple[str, str]] = []

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handle), follow_redirects=False)

    # ------------------------------------------------------------------ pages
    def _idp_form(self, error: str = "") -> str:
        captcha = '<input type="text" name="captcha">' if self.captcha else ""
        return (
            "<html><title>Log in | IU</title><body>"
            '<form method="POST"><input type="hidden" name="state" value="st1">'
            f'<input type="text" name="username"><input type="password" name="password">{captcha}'
            f'{error}<button type="submit" name="action" value="default">Continue</button></form>'
            "</body></html>"
        )

    def _login_page(self) -> str:
        error = (
            '<div id="loginerrormessage">Invalid login, please try again</div>' if self.native_failed else ""
        )
        link = f'<a href="{MOODLE}/auth/oauth2/login.php?id=1&wantsurl=%2F&sesskey=abc">auth.iu.org</a>'
        return f"<html><title>Log in to the site</title><body>{error}{NATIVE_FORM.format(moodle=MOODLE)}{link}</body></html>"

    # ------------------------------------------------------------------ routing
    def handle(self, request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        form = urllib.parse.parse_qs(request.content.decode()) if request.method == "POST" else {}
        if "password" in form:
            self.password_posts.append((host, path))
        password = form.get("password", [""])[0]

        if host == MOODLE_HOST:
            if path == "/login/token.php":
                if self.token_endpoint_ok and password == GOOD_PASSWORD:
                    return httpx.Response(200, json={"token": TOKEN, "privatetoken": "priv"})
                return httpx.Response(
                    200, json={"error": "Invalid login, please try again", "errorcode": "invalidlogin"}
                )
            if path == "/login/index.php" and request.method == "GET":
                return httpx.Response(200, html=self._login_page())
            if path == "/login/index.php":
                assert form["logintoken"] == ["lt1"]
                if self.native_ok and password == GOOD_PASSWORD:
                    self.logged_in = True
                    return httpx.Response(303, headers={"location": f"{MOODLE}/my/"})
                self.native_failed = True
                return httpx.Response(303, headers={"location": f"{MOODLE}/login/index.php"})
            if path == "/auth/oauth2/login.php":
                return httpx.Response(
                    303, headers={"location": f"https://{self.idp_host}/authorize?state=st1"}
                )
            if path == "/admin/oauth2callback.php":
                self.logged_in = True
                return httpx.Response(303, headers={"location": f"{MOODLE}/my/"})
            if path == "/my/":
                return httpx.Response(200, html="<html><title>Dashboard</title></html>")
            if path == "/admin/tool/mobile/launch.php":
                if not self.logged_in:
                    return httpx.Response(303, headers={"location": f"{MOODLE}/login/index.php"})
                passport = request.url.params["passport"]
                site_hash = hashlib.md5((MOODLE + passport).encode()).hexdigest()
                payload = base64.b64encode(f"{site_hash}:::{TOKEN}:::priv".encode()).decode()
                return httpx.Response(303, headers={"location": f"moodlemobile://token={payload}"})
        if host == self.idp_host:
            if path == "/authorize":
                return httpx.Response(302, headers={"location": "/u/login?state=st1"})
            if path == "/u/login" and request.method == "GET":
                return httpx.Response(200, html=self._idp_form())
            if path == "/u/login":
                assert form["state"] == ["st1"] and form["action"] == ["default"]
                if password != GOOD_PASSWORD:
                    error = '<span class="ulp-input-error-message">Wrong email or password</span>'
                    return httpx.Response(400, html=self._idp_form(error))
                target = "/u/mfa-otp-challenge?state=st1" if self.mfa else "/authorize/resume?state=st1"
                return httpx.Response(302, headers={"location": target})
            if path == "/u/mfa-otp-challenge":
                return httpx.Response(200, html='<form method="POST"><input type="text" name="code"></form>')
            if path == "/authorize/resume":
                return httpx.Response(
                    302, headers={"location": f"{MOODLE}/admin/oauth2callback.php?code=c&state=s"}
                )
        return httpx.Response(404, html="<html><title>Error</title></html>")


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, moodle_url=MOODLE, data_dir=tmp_path / "data", anthropic_enabled=False)


def test_host_allowed():
    suffixes = ["iu.org", "iubh.de"]
    assert host_allowed("https://auth.iu.org/u/login", MOODLE, suffixes)
    assert host_allowed(f"{MOODLE}/login/index.php", MOODLE, suffixes)
    assert not host_allowed("https://auth.iu.org.evil.example/u/login", MOODLE, suffixes)
    assert not host_allowed("https://notiu.org/login", MOODLE, suffixes)
    assert not host_allowed("http://auth.iu.org/u/login", MOODLE, suffixes)


def test_form_finder_ignores_hidden_guest_login():
    html = (
        '<form id="guestlogin"><input type="hidden" name="username" value="guest">'
        '<input type="hidden" name="password" value="guest"></form>' + NATIVE_FORM.format(moodle=MOODLE)
    )
    form = find_login_form(html, f"{MOODLE}/login/index.php")
    assert form is not None and form.fields["logintoken"] == "lt1"
    assert (form.user_field, form.password_field) == ("username", "password")


def test_token_endpoint_login(settings):
    campus = FakeCampus(token_endpoint_ok=True)
    token = password_login(settings, "student@iu.org", GOOD_PASSWORD, http=campus.client())
    assert token.token == TOKEN
    assert campus.password_posts == [(MOODLE_HOST, "/login/token.php")]


def test_sso_login_when_the_token_endpoint_refuses(settings):
    campus = FakeCampus()
    messages: list[str] = []
    token = password_login(
        settings, "student@iu.org", GOOD_PASSWORD, http=campus.client(), log=messages.append
    )
    assert token.token == TOKEN
    assert token.site_hash_valid is True
    assert [host for host, _ in campus.password_posts] == [MOODLE_HOST, "auth.iu.org"]
    assert any("auth.iu.org" in message for message in messages)


def test_wrong_password_stops_after_the_identity_provider(settings):
    campus = FakeCampus(native_ok=True)
    with pytest.raises(LoginError) as info:
        password_login(settings, "student@iu.org", "wrong", http=campus.client())
    assert info.value.code == "invalidlogin"
    assert "Wrong email or password" in str(info.value)
    assert (MOODLE_HOST, "/login/index.php") not in campus.password_posts


def test_password_never_goes_to_an_unknown_host(settings):
    campus = FakeCampus(idp_host="login.evil.example")
    with pytest.raises(LoginError):
        password_login(settings, "student@iu.org", GOOD_PASSWORD, http=campus.client())
    assert {host for host, _ in campus.password_posts} <= {MOODLE_HOST}


def test_second_factor_needs_the_browser(settings):
    campus = FakeCampus(mfa=True)
    with pytest.raises(InteractiveLoginRequired):
        password_login(settings, "student@iu.org", GOOD_PASSWORD, http=campus.client())


def test_captcha_is_never_answered(settings):
    campus = FakeCampus(captcha=True)
    with pytest.raises(InteractiveLoginRequired):
        password_login(settings, "student@iu.org", GOOD_PASSWORD, http=campus.client())
    assert "auth.iu.org" not in {host for host, _ in campus.password_posts}


def test_native_form_login(settings):
    campus = FakeCampus(native_ok=True)
    token = password_login(settings, "student", GOOD_PASSWORD, method="form", http=campus.client())
    assert token.token == TOKEN
    assert campus.password_posts == [(MOODLE_HOST, "/login/index.php")]


def test_native_form_reports_the_moodle_error(settings):
    campus = FakeCampus()
    with pytest.raises(LoginError) as info:
        password_login(settings, "student", "wrong", method="form", http=campus.client())
    assert info.value.code == "invalidlogin"
    assert "Invalid login" in str(info.value)

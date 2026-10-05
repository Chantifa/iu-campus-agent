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
    find_skip_form,
    host_allowed,
    password_login,
)

MOODLE = "https://mycampus-classic.iu.org"
MOODLE_HOST = "mycampus-classic.iu.org"
IDP_HOST = "auth.iu.org"
TOKEN = "b" * 32
GOOD_PASSWORD = "correct horse"
USER = "student@iu.org"

NATIVE_FORM = (
    '<form action="{moodle}/login/index.php" method="post" id="login">'
    '<input type="hidden" name="anchor" value=""><input type="hidden" name="logintoken" value="lt1">'
    '<input type="text" name="username"><input type="password" name="password">'
    '<button type="submit" id="loginbtn">Log in</button></form>'
)
PASSKEY_PAGE = (
    "<html><title>Create a passkey | IU</title><body><h1>Sign in faster</h1>"
    '<form method="POST"><input type="hidden" name="state" value="st1">'
    '<button type="submit" name="action" value="default">Create a passkey</button>'
    '<button type="submit" name="action" value="abort-passkey-enrollment">Continue without passkeys</button>'
    "</form></body></html>"
)
VERIFY_PAGE = (
    "<html><title>Verify your e-mail | IU</title><body><h1>Check your inbox</h1>"
    '<form method="POST"><input type="hidden" name="state" value="st1">'
    '<button type="submit" name="action" value="resend">Resend e-mail</button></form></body></html>'
)


class FakeCampus:
    """A Moodle site plus an Auth0-style identity provider behind one mock transport."""

    def __init__(
        self,
        *,
        sso: bool = True,
        token_endpoint_ok: bool = False,
        token_error: str = "invalidlogin",
        native_ok: bool = False,
        idp_host: str = IDP_HOST,
        after_password: str = "resume",  # resume | mfa | passkey | verify
        captcha: bool = False,
        moodle_rejects: bool = False,
    ) -> None:
        self.sso = sso
        self.token_endpoint_ok = token_endpoint_ok
        self.token_error = token_error
        self.native_ok = native_ok
        self.idp_host = idp_host
        self.after_password = after_password
        self.captcha = captcha
        self.moodle_rejects = moodle_rejects
        self.logged_in = False
        self.login_error = ""
        self.posts: list[tuple[str, str]] = []
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
        error = f'<div id="loginerrormessage">{self.login_error}</div>' if self.login_error else ""
        link = (
            f'<a href="{MOODLE}/auth/oauth2/login.php?id=1&wantsurl=%2F&sesskey=abc">auth.iu.org</a>'
            if self.sso
            else ""
        )
        form = NATIVE_FORM.format(moodle=MOODLE)
        return f"<html><title>Log in to the site</title><body>{error}{form}{link}</body></html>"

    def _redirect(self, target: str, status: int = 302) -> httpx.Response:
        return httpx.Response(status, headers={"location": target})

    # ------------------------------------------------------------------ routing
    def handle(self, request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        form = urllib.parse.parse_qs(request.content.decode()) if request.method == "POST" else {}
        if request.method == "POST":
            self.posts.append((host, path))
        if "password" in form:
            self.password_posts.append((host, path))
        password = form.get("password", [""])[0]

        if host == MOODLE_HOST:
            return self._moodle(request, path, form, password)
        if host == self.idp_host:
            return self._idp(request, path, form, password)
        return httpx.Response(404, html="<html><title>Error</title></html>")

    def _moodle(self, request: httpx.Request, path: str, form: dict, password: str) -> httpx.Response:
        if path == "/lib/ajax/service-nologin.php":
            providers = (
                [{"name": "auth.iu.org", "url": f"{MOODLE}/auth/oauth2/login.php?id=1"}] if self.sso else []
            )
            return httpx.Response(200, json=[{"error": False, "data": {"identityproviders": providers}}])
        if path == "/login/token.php":
            if self.token_endpoint_ok and password == GOOD_PASSWORD:
                return httpx.Response(200, json={"token": TOKEN, "privatetoken": "priv"})
            return httpx.Response(200, json={"error": "Refused by the fake", "errorcode": self.token_error})
        if path == "/login/index.php" and request.method == "GET":
            return httpx.Response(200, html=self._login_page())
        if path == "/login/index.php":
            assert form["logintoken"] == ["lt1"]
            if self.native_ok and password == GOOD_PASSWORD:
                self.logged_in = True
                return self._redirect(f"{MOODLE}/my/", 303)
            self.login_error = "Invalid login, please try again"
            return self._redirect(f"{MOODLE}/login/index.php", 303)
        if path == "/auth/oauth2/login.php":
            return self._redirect(f"https://{self.idp_host}/authorize?state=st1", 303)
        if path == "/admin/oauth2callback.php":
            if self.moodle_rejects:
                self.login_error = "The login attempt failed: this account is not linked"
                return self._redirect(f"{MOODLE}/login/index.php", 303)
            self.logged_in = True
            return self._redirect(f"{MOODLE}/my/", 303)
        if path == "/my/":
            return httpx.Response(200, html="<html><title>Dashboard</title></html>")
        if path == "/admin/tool/mobile/launch.php":
            if not self.logged_in:
                return self._redirect(f"{MOODLE}/login/index.php", 303)
            passport = request.url.params["passport"]
            site_hash = hashlib.md5((MOODLE + passport).encode()).hexdigest()
            payload = base64.b64encode(f"{site_hash}:::{TOKEN}:::priv".encode()).decode()
            return self._redirect(f"moodlemobile://token={payload}", 303)
        return httpx.Response(404, html="<html><title>Error</title></html>")

    def _idp(self, request: httpx.Request, path: str, form: dict, password: str) -> httpx.Response:
        if path == "/authorize":
            return self._redirect("/u/login?state=st1")
        if path == "/u/login" and request.method == "GET":
            return httpx.Response(200, html=self._idp_form())
        if path == "/u/login":
            assert form["state"] == ["st1"] and form["action"] == ["default"]
            if password != GOOD_PASSWORD:
                error = '<span class="ulp-input-error-message">Wrong email or password</span>'
                return httpx.Response(400, html=self._idp_form(error))
            target = {
                "resume": "/authorize/resume?state=st1",
                "mfa": "/u/mfa-otp-challenge?state=st1",
                "passkey": "/u/passkey-enrollment?state=st1",
                "verify": "/u/email-verification?state=st1",
            }[self.after_password]
            return self._redirect(target)
        if path == "/u/mfa-otp-challenge":
            return httpx.Response(200, html='<form method="POST"><input type="text" name="code"></form>')
        if path == "/u/passkey-enrollment" and request.method == "GET":
            return httpx.Response(200, html=PASSKEY_PAGE)
        if path == "/u/passkey-enrollment":
            assert form["action"] == ["abort-passkey-enrollment"] and "password" not in form
            return self._redirect("/authorize/resume?state=st1")
        if path == "/u/email-verification":
            return httpx.Response(200, html=VERIFY_PAGE)
        if path == "/authorize/resume":
            return self._redirect(f"{MOODLE}/admin/oauth2callback.php?code=c0de&state=s")
        return httpx.Response(404, html="<html><title>Error</title></html>")


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, moodle_url=MOODLE, data_dir=tmp_path / "data", anthropic_enabled=False)


# ----------------------------------------------------------------------------- units
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


def test_skip_form_only_matches_optional_prompts():
    skip = find_skip_form(PASSKEY_PAGE, "https://auth.iu.org/u/passkey-enrollment?state=st1")
    assert skip is not None
    assert skip.fields == {"state": "st1", "action": "abort-passkey-enrollment"}
    assert find_skip_form(VERIFY_PAGE, "https://auth.iu.org/u/email-verification") is None


# ----------------------------------------------------------------------------- single sign-on
def test_sso_site_sends_the_password_to_the_identity_provider_only(settings):
    campus = FakeCampus()
    messages: list[str] = []
    token = password_login(settings, USER, GOOD_PASSWORD, http=campus.client(), log=messages.append)
    assert token.token == TOKEN
    assert token.site_hash_valid is True
    assert campus.password_posts == [(IDP_HOST, "/u/login")]
    assert any("single sign-on" in message for message in messages)


def test_wrong_password_is_reported_by_the_identity_provider_and_not_retried(settings):
    campus = FakeCampus(native_ok=True, token_endpoint_ok=True)
    with pytest.raises(LoginError) as info:
        password_login(settings, USER, "wrong", http=campus.client())
    assert info.value.code == "invalidlogin"
    assert "auth.iu.org did not accept" in str(info.value)
    assert "Wrong email or password" in str(info.value)
    assert campus.password_posts == [(IDP_HOST, "/u/login")]


def test_password_never_goes_to_an_unknown_host(settings):
    campus = FakeCampus(idp_host="login.evil.example")
    with pytest.raises(LoginError):
        password_login(settings, USER, GOOD_PASSWORD, http=campus.client())
    assert {host for host, _ in campus.password_posts} <= {MOODLE_HOST}


def test_second_factor_needs_the_browser(settings):
    campus = FakeCampus(after_password="mfa")
    with pytest.raises(InteractiveLoginRequired):
        password_login(settings, USER, GOOD_PASSWORD, http=campus.client())
    assert campus.password_posts == [(IDP_HOST, "/u/login")]


def test_captcha_is_never_answered(settings):
    campus = FakeCampus(captcha=True)
    with pytest.raises(InteractiveLoginRequired):
        password_login(settings, USER, GOOD_PASSWORD, http=campus.client())
    assert campus.password_posts == []


def test_optional_passkey_prompt_is_skipped(settings):
    campus = FakeCampus(after_password="passkey")
    messages: list[str] = []
    token = password_login(settings, USER, GOOD_PASSWORD, http=campus.client(), log=messages.append)
    assert token.token == TOKEN
    assert any("Continue without passkeys" in message for message in messages)
    assert campus.password_posts == [(IDP_HOST, "/u/login")]


def test_unknown_page_after_the_password_is_explained_without_fallback(settings):
    campus = FakeCampus(after_password="verify", native_ok=True, token_endpoint_ok=True)
    with pytest.raises(InteractiveLoginRequired) as info:
        password_login(settings, USER, GOOD_PASSWORD, http=campus.client())
    message = str(info.value)
    assert "accepted the password" in message
    assert "Verify your e-mail" in message and "Resend e-mail" in message
    assert campus.password_posts == [(IDP_HOST, "/u/login")]
    assert campus.posts == [(IDP_HOST, "/u/login")]  # the unknown button was not pressed


def test_error_on_the_moodle_side_is_reported(settings):
    campus = FakeCampus(moodle_rejects=True, native_ok=True)
    with pytest.raises(LoginError) as info:
        password_login(settings, USER, GOOD_PASSWORD, http=campus.client())
    assert info.value.code == "notloggedin"
    assert "this account is not linked" in str(info.value)
    assert campus.password_posts == [(IDP_HOST, "/u/login")]


def test_debug_trace_shows_the_route_but_no_secrets(settings):
    campus = FakeCampus()
    lines: list[str] = []
    password_login(settings, USER, GOOD_PASSWORD, http=campus.client(), log=lines.append, debug=True)
    trace = "\n".join(lines)
    assert "POST auth.iu.org/u/login -> 302 -> auth.iu.org/authorize/resume" in trace
    assert "moodlemobile://token=<hidden>" in trace
    assert 'GET auth.iu.org/u/login -> 200  "Log in | IU"' in trace
    for secret in (GOOD_PASSWORD, TOKEN, "st1", "c0de", "sesskey", "abc"):
        assert secret not in trace


# ----------------------------------------------------------------------------- sites without SSO
def test_site_without_sso_uses_the_token_endpoint(settings):
    campus = FakeCampus(sso=False, token_endpoint_ok=True)
    token = password_login(settings, "student", GOOD_PASSWORD, http=campus.client())
    assert token.token == TOKEN
    assert campus.password_posts == [(MOODLE_HOST, "/login/token.php")]


def test_site_without_sso_stops_when_moodle_rejects_the_password(settings):
    campus = FakeCampus(sso=False, native_ok=True)
    with pytest.raises(LoginError) as info:
        password_login(settings, "student", "wrong", http=campus.client())
    assert info.value.code == "invalidlogin"
    assert campus.password_posts == [(MOODLE_HOST, "/login/token.php")]


def test_site_without_sso_falls_back_to_the_form_when_tokens_are_unavailable(settings):
    campus = FakeCampus(sso=False, native_ok=True, token_error="servicenotavailable")
    token = password_login(settings, "student", GOOD_PASSWORD, http=campus.client())
    assert token.token == TOKEN
    assert campus.password_posts == [(MOODLE_HOST, "/login/token.php"), (MOODLE_HOST, "/login/index.php")]


def test_native_form_login_and_its_error(settings):
    campus = FakeCampus(native_ok=True)
    token = password_login(settings, "student", GOOD_PASSWORD, method="form", http=campus.client())
    assert token.token == TOKEN
    assert campus.password_posts == [(MOODLE_HOST, "/login/index.php")]

    campus = FakeCampus()
    with pytest.raises(LoginError) as info:
        password_login(settings, "student", "wrong", method="form", http=campus.client())
    assert info.value.code == "invalidlogin"
    assert "Invalid login" in str(info.value)

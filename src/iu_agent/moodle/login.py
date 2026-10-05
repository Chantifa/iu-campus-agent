"""Username + password login for Moodle sites, without a browser.

:func:`password_login` tries these strategies (``method="auto"``):

1. ``token`` - Moodle's own token endpoint ``login/token.php``. Works when Moodle itself checks the
   password (manual, LDAP, database ... authentication).
2. ``sso``   - sign in through the site's single sign-on provider: open the Moodle login page,
   follow its SSO link to the identity provider, submit the provider's HTML login form (IU uses
   Auth0 at ``auth.iu.org``) and follow the redirects back to Moodle.
3. ``form``  - Moodle's own HTML login form (``login/index.php``).

After 2. or 3. the session is logged in and the web-service token is taken from the redirect of
``admin/tool/mobile/launch.php`` (``moodlemobile://token=...``), the address the official app
receives and a desktop browser usually hides.

The password is only posted over HTTPS to the Moodle host or to hosts matching
``MOODLE_SSO_HOSTS``; it is never stored or logged. CAPTCHAs and multi-factor prompts are not
handled, they raise :class:`InteractiveLoginRequired` (use ``iu-agent moodle login --browser``).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from iu_agent.config import Settings
from iu_agent.moodle.auth import MoodleToken, make_passport, parse_launch_response

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36"
)
URL_SCHEME = "moodlemobile"
MAX_HOPS = 15
METHODS = ("auto", "token", "sso", "form")

_USER_FIELDS = ("username", "email", "identifier", "login", "user", "j_username", "loginfmt")
_CAPTCHA_FIELDS = ("captcha", "g-recaptcha-response", "h-captcha-response", "cf-turnstile-response")
_CODE_FIELDS = ("code", "otp", "totp", "mfa", "verification_code", "passcode", "sms_code")
_SSO_LINK_HINTS = ("auth/oauth2/login.php", "auth/oidc/", "auth/saml2/login.php", "auth/shibboleth/")
_ERROR_SELECTORS = (
    "#loginerrormessage",
    ".loginerrors",
    ".ulp-input-error-message",
    "[id^=error-element-]",
    "#prompt-alert",
    ".alert-danger",
    "[role=alert]",
)

Log = Callable[[str], None]


class LoginError(RuntimeError):
    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class InteractiveLoginRequired(LoginError):
    """The site asks for something a script must not answer (CAPTCHA, second factor)."""


@dataclass
class LoginForm:
    action: str
    fields: dict[str, str] = field(default_factory=dict)
    user_field: str | None = None
    password_field: str | None = None
    challenge: str | None = None


# ----------------------------------------------------------------------------- helpers
def sso_host_suffixes(settings: Settings) -> list[str]:
    return [h.strip().lower().lstrip(".") for h in settings.moodle_sso_hosts.split(",") if h.strip()]


def host_allowed(url: str, base_url: str, suffixes: list[str]) -> bool:
    """True when credentials may travel to ``url``: HTTPS, and the Moodle host or an SSO host."""
    target, base = urlparse(url), urlparse(base_url)
    host, base_host = (target.hostname or "").lower(), (base.hostname or "").lower()
    if target.scheme != "https" and not (target.scheme == base.scheme and host == base_host):
        return False
    if host == base_host:
        return True
    return any(host == suffix or host.endswith("." + suffix) for suffix in suffixes)


def new_http_client(timeout: float = 60.0) -> httpx.Client:
    return httpx.Client(timeout=timeout, follow_redirects=False, headers={"User-Agent": BROWSER_UA})


def find_login_form(html: str, page_url: str, *, identifier_only: bool = False) -> LoginForm | None:
    """The form with a visible password input (or, optionally, a user-name-only first step)."""
    soup = BeautifulSoup(html, "html.parser")
    for form in soup.find_all("form"):
        inputs = form.find_all("input")
        password_input = next(
            (i for i in inputs if (i.get("type") or "").lower() == "password" and i.get("name")), None
        )
        user_input = next(
            (
                i
                for i in inputs
                if (i.get("type") or "text").lower() in ("text", "email")
                and (i.get("name") or "").lower() in _USER_FIELDS
            ),
            None,
        )
        if password_input is None and not (identifier_only and user_input is not None):
            continue
        fields: dict[str, str] = {}
        challenge: str | None = None
        for element in [*inputs, *form.find_all("textarea")]:
            name = element.get("name")
            if not name:
                continue
            kind = (element.get("type") or "text").lower()
            lowered = name.lower()
            if lowered in _CAPTCHA_FIELDS:
                challenge = "a CAPTCHA"
            elif kind != "hidden" and lowered in _CODE_FIELDS:
                challenge = "a verification code"
            if kind in ("submit", "button", "image", "file", "reset"):
                continue
            if kind in ("checkbox", "radio") and not element.has_attr("checked"):
                continue
            fields[name] = element.get("value") or ""
        submit = next(
            (b for b in form.find_all("button") if b.get("name") and (b.get("type") or "submit") == "submit"),
            None,
        ) or next((i for i in inputs if (i.get("type") or "").lower() == "submit" and i.get("name")), None)
        if submit is not None:
            fields[submit["name"]] = submit.get("value") or ""
        return LoginForm(
            action=urljoin(page_url, form.get("action") or page_url),
            fields=fields,
            user_field=user_input.get("name") if user_input is not None else None,
            password_field=password_input.get("name") if password_input is not None else None,
            challenge=challenge,
        )
    return None


def find_challenge(html: str, page_url: str) -> str | None:
    """A second-factor or CAPTCHA page that is not a login form."""
    path = urlparse(page_url).path.lower()
    soup = BeautifulSoup(html, "html.parser")
    for element in [*soup.find_all("input"), *soup.find_all("textarea")]:
        name = (element.get("name") or "").lower()
        kind = (element.get("type") or "text").lower()
        if name in _CAPTCHA_FIELDS:
            return "a CAPTCHA"
        if kind != "hidden" and name in _CODE_FIELDS:
            return "a verification code"
    if any(hint in path for hint in ("/mfa", "captcha", "/otp", "challenge")):
        return "an additional verification step"
    return None


def find_sso_link(html: str, page_url: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for anchor in soup.find_all("a"):
        href = anchor.get("href") or ""
        if any(hint in href for hint in _SSO_LINK_HINTS):
            return urljoin(page_url, href)
    return None


def extract_error(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for selector in _ERROR_SELECTORS:
        for element in soup.select(selector):
            text = element.get_text(" ", strip=True)
            if text:
                return text[:200]
    return None


def _follow(
    http: httpx.Client, response: httpx.Response, base_url: str, suffixes: list[str]
) -> httpx.Response:
    """Follow redirects by hand so every hop is checked against the allowed hosts."""
    for _ in range(MAX_HOPS):
        if not response.is_redirect:
            return response
        target = urljoin(str(response.url), response.headers.get("location", ""))
        if urlparse(target).scheme not in ("http", "https"):
            return response  # custom scheme, e.g. moodlemobile://token=...
        if response.status_code in (307, 308) and response.request.method != "GET":
            raise LoginError(
                "The login was redirected in a way that would resend the password.", "unsafe_redirect"
            )
        if not host_allowed(target, base_url, suffixes):
            raise LoginError(
                f"Unexpected redirect to {urlparse(target).hostname}. If that host belongs to your "
                "university's login, add it to MOODLE_SSO_HOSTS.",
                "unexpected_host",
            )
        response = http.get(target)
    raise LoginError("Too many redirects during the login.", "redirect_loop")


# ----------------------------------------------------------------------------- strategies
def token_via_endpoint(
    http: httpx.Client, base_url: str, username: str, password: str, service: str
) -> MoodleToken:
    response = http.post(
        f"{base_url}/login/token.php",
        data={"username": username, "password": password, "service": service},
    )
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as exc:
        raise LoginError("The token endpoint did not return JSON.", "notjson") from exc
    if payload.get("token"):
        return MoodleToken(token=payload["token"], private_token=payload.get("privatetoken"))
    raise LoginError(payload.get("error") or "The token request was refused.", payload.get("errorcode"))


def _submit_credentials(
    http: httpx.Client,
    page: httpx.Response,
    base_url: str,
    username: str,
    password: str,
    suffixes: list[str],
    log: Log,
) -> httpx.Response:
    for _ in range(4):  # user-name-first providers need two steps
        form = find_login_form(page.text, str(page.url), identifier_only=True)
        if form is None:
            return page
        if form.challenge:
            raise InteractiveLoginRequired(f"The login page asks for {form.challenge}.", "challenge")
        if not host_allowed(form.action, base_url, suffixes):
            raise LoginError(
                f"The login form posts to {urlparse(form.action).hostname}, which is not an allowed host "
                "(MOODLE_SSO_HOSTS); the password was not sent.",
                "unexpected_host",
            )
        data = dict(form.fields)
        if form.user_field:
            data[form.user_field] = username
        if form.password_field:
            data[form.password_field] = password
        log(f"signing in at {urlparse(form.action).hostname}")
        page = _follow(http, http.post(form.action, data=data), base_url, suffixes)
        if not form.password_field:
            continue  # the next page asks for the password
        challenge = find_challenge(page.text, str(page.url))
        if challenge:
            raise InteractiveLoginRequired(f"The site asks for {challenge} after the password.", "challenge")
        again = find_login_form(page.text, str(page.url), identifier_only=True)
        if again is not None and urlparse(again.action).hostname == urlparse(form.action).hostname:
            raise LoginError(
                extract_error(page.text) or "The user name or password was not accepted.", "invalidlogin"
            )
        return page
    raise LoginError("The login form kept coming back.", "loop")


def login_via_sso(
    http: httpx.Client, base_url: str, username: str, password: str, suffixes: list[str], log: Log
) -> None:
    page = _follow(http, http.get(f"{base_url}/login/index.php"), base_url, suffixes)
    link = find_sso_link(page.text, str(page.url))
    if link is not None:
        page = _follow(http, http.get(link), base_url, suffixes)
    form = find_login_form(page.text, str(page.url), identifier_only=True)
    base_host = urlparse(base_url).hostname
    if form is None or (link is None and urlparse(form.action).hostname == base_host):
        raise LoginError("The login page offers no single sign-on provider.", "nosso")
    _submit_credentials(http, page, base_url, username, password, suffixes, log)


def login_via_form(
    http: httpx.Client, base_url: str, username: str, password: str, suffixes: list[str], log: Log
) -> None:
    page = _follow(http, http.get(f"{base_url}/login/index.php"), base_url, suffixes)
    form = find_login_form(page.text, str(page.url))
    if form is None or urlparse(form.action).hostname != urlparse(base_url).hostname:
        raise LoginError("The site has no user name / password form of its own.", "noform")
    _submit_credentials(http, page, base_url, username, password, suffixes, log)


def token_from_session(http: httpx.Client, base_url: str, service: str, suffixes: list[str]) -> MoodleToken:
    """Ask the logged-in session for the app token and read it from the redirect."""
    passport = make_passport()
    response = http.get(
        f"{base_url}/admin/tool/mobile/launch.php",
        params={"service": service, "passport": passport, "urlscheme": URL_SCHEME},
    )
    for _ in range(MAX_HOPS):
        location = response.headers.get("location", "") if response.is_redirect else ""
        if location.startswith(f"{URL_SCHEME}://"):
            return parse_launch_response(location, base_url, passport)
        if response.is_redirect:
            target = urljoin(str(response.url), location)
            if urlparse(target).path.rstrip("/").endswith("/login/index.php"):
                raise LoginError(
                    "Moodle did not accept the login (the session is not signed in).", "notloggedin"
                )
            if not host_allowed(target, base_url, suffixes):
                raise LoginError(f"Unexpected redirect to {urlparse(target).hostname}.", "unexpected_host")
            response = http.get(target)
            continue
        match = re.search(rf"{URL_SCHEME}://token=[A-Za-z0-9+/=%_\-]+", response.text)
        if match:
            return parse_launch_response(match.group(0), base_url, passport)
        title = BeautifulSoup(response.text, "html.parser").find("title")
        shown = title.get_text(strip=True)[:80] if title else f"HTTP {response.status_code}"
        raise LoginError(
            f"Moodle showed a page instead of the token ({shown}). Open myCampus in a browser once, "
            "complete what it asks for (policy, profile), then retry.",
            "needsattention",
        )
    raise LoginError("Too many redirects while requesting the token.", "redirect_loop")


# ----------------------------------------------------------------------------- entry point
def password_login(
    settings: Settings,
    username: str,
    password: str,
    *,
    method: str = "auto",
    http: httpx.Client | None = None,
    log: Log | None = None,
) -> MoodleToken:
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    log = log or (lambda _message: None)
    base_url = settings.moodle_url.rstrip("/")
    suffixes = sso_host_suffixes(settings)
    own_client = http is None
    http = http or new_http_client()
    try:
        if method in ("auto", "token"):
            try:
                token = token_via_endpoint(http, base_url, username, password, settings.moodle_service)
                log("the token endpoint accepted the credentials")
                return token
            except LoginError as exc:
                if method == "token":
                    raise
                log(f"token endpoint: {exc} - trying the web login")
        strategies = {"auto": ("sso", "form"), "sso": ("sso",), "form": ("form",)}.get(method, ())
        failures: list[LoginError] = []
        for strategy in strategies:
            runner = login_via_sso if strategy == "sso" else login_via_form
            try:
                runner(http, base_url, username, password, suffixes, log)
                log("signed in, requesting the app token")
                return token_from_session(http, base_url, settings.moodle_service, suffixes)
            except InteractiveLoginRequired:
                raise
            except LoginError as exc:
                if exc.code == "invalidlogin":
                    raise  # wrong credentials: do not repeat them against another form
                failures.append(exc)
                http.cookies.clear()
        if failures:
            raise LoginError("; ".join(str(f) for f in failures), failures[-1].code)
        raise LoginError("The login failed.", "failed")
    finally:
        if own_client:
            http.close()

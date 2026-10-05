"""User name + password login for Moodle sites, without a browser.

:func:`password_login` signs in with one of these strategies:

* ``sso``   - sign in through the site's single sign-on provider: open the Moodle login page,
  follow its SSO link to the identity provider, submit the provider's HTML login form (IU uses
  Auth0 at ``auth.iu.org``) and follow the redirects back to Moodle.
* ``token`` - Moodle's own token endpoint ``login/token.php``. Works when Moodle itself checks the
  password (manual, LDAP, database ... authentication).
* ``form``  - Moodle's own HTML login form (``login/index.php``).

``method="auto"`` uses single sign-on when the site announces an identity provider and the
token endpoint (then the form) otherwise. Credentials that one system has rejected are never
repeated against another one.

After ``sso`` or ``form`` the session is signed in and the web-service token is read from the
redirect of ``admin/tool/mobile/launch.php`` (``moodlemobile://token=...``), the address the
official app receives and a desktop browser usually hides.

The password is only posted over HTTPS to the Moodle host or to hosts matching
``MOODLE_SSO_HOSTS``; it is never stored or logged. CAPTCHAs and second factors are never
answered: they raise :class:`InteractiveLoginRequired` (use ``iu-agent moodle login --browser``).
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
from iu_agent.moodle.client import MoodleClient, MoodleError

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36"
)
URL_SCHEME = "moodlemobile"
MAX_HOPS = 15
METHODS = ("auto", "sso", "token", "form")

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
# Optional prompts an identity provider may show after the password ("create a passkey?" ...).
_SKIP_VALUES = {
    "abort-passkey-enrollment",
    "snooze-enrollment",
    "skip",
    "skip-enrollment",
    "remind-later",
    "refuse-add-device",
}
_SKIP_TEXT_RE = re.compile(
    r"continue without|not now|skip|remind me later|maybe later|ohne .{0,30}fortfahren|sp[aä]ter|"
    r"[üu]berspringen|nicht jetzt",
    re.IGNORECASE,
)

Log = Callable[[str], None]


class LoginError(RuntimeError):
    def __init__(self, message: str, code: str | None = None, *, after_password: bool = False) -> None:
        super().__init__(message)
        self.code = code
        # True once a system has seen the password: never retry it somewhere else after that.
        self.after_password = after_password


class InteractiveLoginRequired(LoginError):
    """The site asks for something a script must not answer (CAPTCHA, second factor, a prompt)."""


@dataclass
class LoginForm:
    action: str
    fields: dict[str, str] = field(default_factory=dict)
    user_field: str | None = None
    password_field: str | None = None
    challenge: str | None = None
    user_label: str | None = None  # the site's own wording, e.g. "Personal E-Mail or Username"


@dataclass
class SkipForm:
    action: str
    fields: dict[str, str]
    label: str


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


def _host(url: str) -> str:
    return urlparse(url).hostname or ""


def _title(html: str) -> str | None:
    title = BeautifulSoup(html, "html.parser").find("title")
    return title.get_text(" ", strip=True)[:80] if title else None


def trace_hook(log: Log) -> Callable[[httpx.Response], None]:
    """Log every request as ``METHOD host/path -> status``. Query strings (state, code, sesskey),
    cookies, bodies and the token are never printed."""

    def hook(response: httpx.Response) -> None:
        request = response.request
        line = f"{request.method} {request.url.host}{request.url.path} -> {response.status_code}"
        location = response.headers.get("location")
        if location:
            target = urlparse(urljoin(str(request.url), location))
            if target.scheme in ("http", "https"):
                line += f" -> {target.netloc}{target.path}"
            else:
                line += f" -> {target.scheme}://token=<hidden>"
        elif "html" in response.headers.get("content-type", ""):
            response.read()
            title = _title(response.text)
            if title:
                line += f'  "{title}"'
        log("trace: " + line)

    return hook


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
            user_label=_field_label(soup, user_input) if user_input is not None else None,
        )
    return None


def _field_label(soup: BeautifulSoup, element) -> str | None:
    """The visible label of an input: ``<label for=id>``, a wrapping label, or the placeholder."""
    label = None
    if element.get("id"):
        label = soup.find("label", attrs={"for": element["id"]})
    label = label or element.find_parent("label")
    text = label.get_text(" ", strip=True) if label is not None else (element.get("placeholder") or "")
    text = text.strip().rstrip("*").strip()
    return text[:80] or None


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


def find_skip_form(html: str, page_url: str) -> SkipForm | None:
    """A form offering to skip an optional prompt (for example "Continue without passkeys")."""
    soup = BeautifulSoup(html, "html.parser")
    for form in soup.find_all("form"):
        if form.find("input", {"type": "password"}):
            continue
        hidden = {
            i["name"]: i.get("value") or ""
            for i in form.find_all("input")
            if i.get("name") and (i.get("type") or "").lower() == "hidden"
        }
        for button in form.find_all("button"):
            name, value = button.get("name"), button.get("value") or ""
            label = button.get_text(" ", strip=True)
            if name and (value in _SKIP_VALUES or _SKIP_TEXT_RE.search(label)):
                return SkipForm(
                    action=urljoin(page_url, form.get("action") or page_url),
                    fields={**hidden, name: value},
                    label=label or value,
                )
    return None


def describe_page(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    title = _title(html) or "a page without title"
    heading = soup.find(["h1", "h2"])
    heading_text = heading.get_text(" ", strip=True)[:80] if heading else ""
    buttons = [b.get_text(" ", strip=True)[:40] for b in soup.find_all("button") if b.get_text(strip=True)]
    text = f"'{title}'"
    if heading_text and heading_text.lower() not in title.lower():
        text += f" / '{heading_text}'"
    if buttons:
        text += f" (buttons: {', '.join(buttons[:4])})"
    return text


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
                f"Unexpected redirect to {_host(target)}. If that host belongs to your university's "
                "login, add it to MOODLE_SSO_HOSTS.",
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
    code = payload.get("errorcode")
    message = payload.get("error") or "The token request was refused."
    if code == "invalidlogin":
        raise LoginError(
            f"{_host(base_url)} did not accept the user name or password ({message}).",
            code,
            after_password=True,
        )
    raise LoginError(f"Token endpoint: {message}", code)


def _submit_credentials(
    http: httpx.Client,
    page: httpx.Response,
    base_url: str,
    username: str,
    password: str,
    suffixes: list[str],
    log: Log,
) -> httpx.Response:
    """Fill and post the login form(s) on ``page``; returns the page reached afterwards."""
    for _ in range(4):  # user-name-first providers need two steps
        form = find_login_form(page.text, str(page.url), identifier_only=True)
        if form is None:
            return page
        host = _host(form.action)
        if form.challenge:
            raise InteractiveLoginRequired(
                f"The login page of {host} asks for {form.challenge}.", "challenge"
            )
        if not host_allowed(form.action, base_url, suffixes):
            raise LoginError(
                f"The login form posts to {host}, which is not an allowed host (MOODLE_SSO_HOSTS); "
                "the password was not sent.",
                "unexpected_host",
            )
        data = dict(form.fields)
        if form.user_field:
            data[form.user_field] = username
        if form.password_field:
            data[form.password_field] = password
        log(f"signing in at {host}")
        try:
            page = _follow(http, http.post(form.action, data=data), base_url, suffixes)
        except LoginError as exc:
            exc.after_password = exc.after_password or bool(form.password_field)
            raise
        if not form.password_field:
            continue  # the next page asks for the password
        challenge = find_challenge(page.text, str(page.url))
        if challenge:
            raise InteractiveLoginRequired(
                f"{host} asks for {challenge} after the password.", "challenge", after_password=True
            )
        again = find_login_form(page.text, str(page.url), identifier_only=True)
        if again is not None and _host(again.action) == host:
            reason = extract_error(page.text) or "no reason given"
            asked = f' Its form asks for "{form.user_label}".' if form.user_label else ""
            raise LoginError(
                f"{host} did not accept the user name or password ({reason}).{asked}",
                "invalidlogin",
                after_password=True,
            )
        return page
    raise LoginError("The login form kept coming back.", "loop", after_password=True)


def _leave_identity_provider(
    http: httpx.Client, page: httpx.Response, idp_host: str, base_url: str, suffixes: list[str], log: Log
) -> None:
    """After the password: skip optional prompts until the provider hands back to Moodle."""
    for _ in range(5):
        if _host(str(page.url)) != idp_host:
            return
        skip = find_skip_form(page.text, str(page.url))
        if skip is None or not host_allowed(skip.action, base_url, suffixes):
            raise InteractiveLoginRequired(
                f"{idp_host} accepted the password but shows another page before returning to Moodle: "
                f"{describe_page(page.text)}. Open myCampus in a browser, complete that step once, "
                "then run the login again.",
                "interstitial",
                after_password=True,
            )
        log(f"skipping the optional prompt at {idp_host}: '{skip.label}'")
        page = _follow(http, http.post(skip.action, data=skip.fields), base_url, suffixes)
    raise LoginError(f"{idp_host} kept showing extra pages.", "loop", after_password=True)


def login_via_sso(
    http: httpx.Client, base_url: str, username: str, password: str, suffixes: list[str], log: Log
) -> None:
    page = _follow(http, http.get(f"{base_url}/login/index.php"), base_url, suffixes)
    link = find_sso_link(page.text, str(page.url))
    if link is not None:
        page = _follow(http, http.get(link), base_url, suffixes)
    form = find_login_form(page.text, str(page.url), identifier_only=True)
    base_host = _host(base_url)
    if form is None or (link is None and _host(form.action) == base_host):
        raise LoginError("The login page offers no single sign-on provider.", "nosso")
    idp_host = _host(form.action)
    page = _submit_credentials(http, page, base_url, username, password, suffixes, log)
    _leave_identity_provider(http, page, idp_host, base_url, suffixes, log)


def login_via_form(
    http: httpx.Client, base_url: str, username: str, password: str, suffixes: list[str], log: Log
) -> None:
    page = _follow(http, http.get(f"{base_url}/login/index.php"), base_url, suffixes)
    form = find_login_form(page.text, str(page.url))
    if form is None or _host(form.action) != _host(base_url):
        raise LoginError("The site has no user name / password form of its own.", "noform")
    _submit_credentials(http, page, base_url, username, password, suffixes, log)


def token_from_session(http: httpx.Client, base_url: str, service: str, suffixes: list[str]) -> MoodleToken:
    """Ask the signed-in session for the app token and read it from the redirect."""
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
                reason = None
                if host_allowed(target, base_url, suffixes):
                    reason = extract_error(http.get(target).text)
                detail = f": {reason}" if reason else " (the session is not signed in)."
                raise LoginError(f"Moodle did not complete the sign-in{detail}", "notloggedin")
            if not host_allowed(target, base_url, suffixes):
                raise LoginError(f"Unexpected redirect to {_host(target)}.", "unexpected_host")
            response = http.get(target)
            continue
        match = re.search(rf"{URL_SCHEME}://token=[A-Za-z0-9+/=%_\-]+", response.text)
        if match:
            return parse_launch_response(match.group(0), base_url, passport)
        raise LoginError(
            f"Moodle showed a page instead of the token ({_title(response.text) or response.status_code}). "
            "Open myCampus in a browser once, complete what it asks for (policy, profile), then retry.",
            "needsattention",
        )
    raise LoginError("Too many redirects while requesting the token.", "redirect_loop")


def describe_login(settings: Settings, http: httpx.Client | None = None) -> tuple[str, str | None] | None:
    """``(host, label of the user-name field)`` of the page that will ask for the credentials.

    Nothing is submitted. Returns ``None`` when the page cannot be determined.
    """
    base_url = settings.moodle_url.rstrip("/")
    suffixes = sso_host_suffixes(settings)
    own_client = http is None
    http = http or new_http_client(timeout=30.0)
    try:
        page = _follow(http, http.get(f"{base_url}/login/index.php"), base_url, suffixes)
        link = find_sso_link(page.text, str(page.url))
        if link is not None:
            page = _follow(http, http.get(link), base_url, suffixes)
        form = find_login_form(page.text, str(page.url), identifier_only=True)
        return (_host(form.action), form.user_label) if form is not None else None
    except (LoginError, httpx.HTTPError):
        return None
    finally:
        http.cookies.clear()
        if own_client:
            http.close()


# ----------------------------------------------------------------------------- entry point
def _strategy_order(method: str, http: httpx.Client, base_url: str, log: Log) -> tuple[str, ...]:
    if method != "auto":
        return (method,)
    try:
        providers = MoodleClient.public_config(base_url, http=http).get("identityproviders") or []
    except MoodleError as exc:
        raise LoginError(str(exc), exc.errorcode) from exc
    if providers:
        names = ", ".join(str(p.get("name")) for p in providers)
        log(f"the site signs in through single sign-on ({names})")
        return ("sso", "token", "form")
    return ("token", "form")


def password_login(
    settings: Settings,
    username: str,
    password: str,
    *,
    method: str = "auto",
    http: httpx.Client | None = None,
    log: Log | None = None,
    debug: bool = False,
) -> MoodleToken:
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    log = log or (lambda _message: None)
    base_url = settings.moodle_url.rstrip("/")
    suffixes = sso_host_suffixes(settings)
    own_client = http is None
    http = http or new_http_client()
    hook = trace_hook(log) if debug else None
    if hook is not None:
        http.event_hooks["response"].append(hook)
    try:
        order = _strategy_order(method, http, base_url, log)
        failures: list[LoginError] = []
        for strategy in order:
            try:
                if strategy == "token":
                    token = token_via_endpoint(http, base_url, username, password, settings.moodle_service)
                    log("the token endpoint accepted the credentials")
                    return token
                runner = login_via_sso if strategy == "sso" else login_via_form
                runner(http, base_url, username, password, suffixes, log)
                log("signed in, requesting the app token")
                try:
                    return token_from_session(http, base_url, settings.moodle_service, suffixes)
                except LoginError as exc:
                    exc.after_password = True
                    raise
            except InteractiveLoginRequired:
                raise  # a CAPTCHA or second factor ends the attempt, no other system is tried
            except LoginError as exc:
                if exc.after_password or len(order) == 1:
                    raise  # never repeat credentials one system has already seen
                failures.append(exc)
                log(f"{strategy}: {exc}")
                http.cookies.clear()
        raise LoginError(
            "; ".join(str(f) for f in failures) or "The login failed.",
            failures[-1].code if failures else "failed",
        )
    finally:
        if hook is not None and hook in http.event_hooks["response"]:
            http.event_hooks["response"].remove(hook)
        if own_client:
            http.close()

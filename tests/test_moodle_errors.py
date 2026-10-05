import httpx
import pytest

from iu_agent.moodle.client import MoodleClient, MoodleError

PORTAL = "https://mycampus.iu.org"


def _html_site(request: httpx.Request) -> httpx.Response:
    # a single-page portal answers every path with its HTML shell and status 200
    return httpx.Response(200, html="<!doctype html><html><body>portal</body></html>")


def test_public_config_explains_a_non_moodle_address():
    http = httpx.Client(transport=httpx.MockTransport(_html_site))
    with pytest.raises(MoodleError) as info:
        MoodleClient.public_config(PORTAL, http=http)
    message = str(info.value)
    assert info.value.errorcode == "notmoodle"
    assert "did not answer like a Moodle site" in message
    assert "mycampus-classic.iu.org" in message
    assert "text/html" in message


def test_web_service_call_explains_a_non_moodle_address():
    http = httpx.Client(transport=httpx.MockTransport(_html_site))
    client = MoodleClient(PORTAL, "token", http=http)
    with pytest.raises(MoodleError) as info:
        client.site_info()
    assert info.value.errorcode == "notmoodle"

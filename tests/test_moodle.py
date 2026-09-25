import base64
import hashlib
import urllib.parse

import httpx
import pytest

from iu_agent.moodle.auth import launch_url, parse_launch_response
from iu_agent.moodle.client import MoodleClient, MoodleError, flatten_params, with_token
from iu_agent.moodle.sync import MoodleSync
from iu_agent.rag.ingest import Ingestor

SITE = "https://mycampus-classic.iu.org"
TOKEN = "a" * 32


def test_flatten_params():
    flat = flatten_params({"courseids": [1, 2], "options": [{"name": "x", "value": True}], "userid": 7})
    assert flat == {
        "courseids[0]": "1",
        "courseids[1]": "2",
        "options[0][name]": "x",
        "options[0][value]": "1",
        "userid": "7",
    }


def test_with_token():
    assert with_token("https://x/pluginfile.php/1/a.pdf", "t") == "https://x/pluginfile.php/1/a.pdf?token=t"
    assert with_token("https://x/f?forcedownload=1", "t") == "https://x/f?forcedownload=1&token=t"


def test_parse_launch_response_variants():
    passport = "123.456"
    site_hash = hashlib.md5((SITE + passport).encode()).hexdigest()
    payload = base64.b64encode(f"{site_hash}:::{TOKEN}:::privtoken".encode()).decode()
    parsed = parse_launch_response(f"moodlemobile://token={urllib.parse.quote(payload)}", SITE, passport)
    assert parsed.token == TOKEN
    assert parsed.private_token == "privtoken"
    assert parsed.site_hash_valid is True

    assert parse_launch_response(TOKEN).token == TOKEN
    assert parse_launch_response(payload, SITE, "other").site_hash_valid is False
    with pytest.raises(ValueError):
        parse_launch_response("definitely not a token!!")


def test_launch_url():
    url = launch_url(SITE, "1.5")
    assert url.startswith(f"{SITE}/admin/tool/mobile/launch.php?")
    assert "service=moodle_mobile_app" in url and "urlscheme=moodlemobile" in url and "passport=1.5" in url


# ----------------------------------------------------------------------------- fake Moodle server
COURSE = {"id": 42, "fullname": "Deep Learning (DLMAIDL01)", "shortname": "DLMAIDL01"}


def _ws_response(function: str, params: dict) -> object:
    if function == "core_webservice_get_site_info":
        return {
            "userid": 9,
            "fullname": "Test Student",
            "username": "student",
            "sitename": "IU",
            "release": "4.x",
        }
    if function == "core_enrol_get_users_courses":
        assert params["userid"] == ["9"]
        return [COURSE]
    if function == "core_course_get_contents":
        return [
            {
                "id": 1,
                "name": "Week 1",
                "summary": "<p>Welcome to the deep learning course, this section covers the basics.</p>",
                "modules": [
                    {
                        "id": 100,
                        "modname": "resource",
                        "name": "Course book",
                        "url": f"{SITE}/mod/resource/view.php?id=100",
                        "contents": [
                            {
                                "type": "file",
                                "filename": "book.md",
                                "fileurl": f"{SITE}/webservice/pluginfile.php/1/mod_resource/content/1/book.md",
                                "filesize": 10,
                                "timemodified": 1700000000,
                            },
                            {
                                "type": "file",
                                "filename": "video.mp4",
                                "fileurl": f"{SITE}/webservice/pluginfile.php/1/mod_resource/content/1/video.mp4",
                                "filesize": 10,
                                "timemodified": 1700000000,
                            },
                        ],
                    },
                    {
                        "id": 101,
                        "modname": "page",
                        "name": "Intro page",
                        "url": f"{SITE}/mod/page/view.php?id=101",
                    },
                    {
                        "id": 102,
                        "modname": "quiz",
                        "name": "Self test",
                        "url": f"{SITE}/mod/quiz/view.php?id=102",
                    },
                    {
                        "id": 103,
                        "modname": "url",
                        "name": "External reading",
                        "description": "<p>Read this paper about transformers before the lecture.</p>",
                        "contents": [
                            {
                                "type": "url",
                                "filename": "External reading",
                                "fileurl": "https://example.org/paper",
                            }
                        ],
                    },
                ],
            }
        ]
    if function == "mod_page_get_pages_by_courses":
        return {
            "pages": [
                {
                    "coursemodule": 101,
                    "name": "Intro page",
                    "intro": "",
                    "content": "<h2>Perceptron</h2><p>A perceptron is a linear classifier.</p>",
                    "timemodified": 1,
                }
            ]
        }
    if function == "mod_assign_get_assignments":
        return {
            "courses": [
                {
                    "id": 42,
                    "assignments": [
                        {
                            "cmid": 200,
                            "name": "Homework 1",
                            "intro": "<p>Implement backpropagation in numpy.</p>",
                            "duedate": 1900000000,
                            "timemodified": 1,
                        }
                    ],
                }
            ]
        }
    if function == "mod_forum_get_forums_by_courses":
        return [
            {"id": 5, "type": "news", "name": "Announcements", "cmid": 300},
            {"id": 6, "type": "general", "name": "Q&A", "cmid": 301},
        ]
    if function == "mod_forum_get_forum_discussions":
        return {
            "discussions": [
                {
                    "discussion": 77,
                    "name": "Exam date",
                    "message": "<p>The exam takes place on 12 December.</p>",
                    "userfullname": "Tutor",
                    "created": 1700000000,
                    "timemodified": 1700000000,
                }
            ]
        }
    return {"exception": "moodle_exception", "errorcode": "invalidfunction", "message": f"unknown {function}"}


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/webservice/rest/server.php"):
        form = urllib.parse.parse_qs(request.content.decode())
        assert form["wstoken"] == [TOKEN]
        function = form["wsfunction"][0]
        params = {k: v for k, v in form.items() if k not in ("wstoken", "wsfunction", "moodlewsrestformat")}
        return httpx.Response(200, json=_ws_response(function, params))
    if "pluginfile.php" in path:
        assert request.url.params.get("token") == TOKEN
        return httpx.Response(
            200, content=b"# Book\n\nDropout regularises networks by randomly disabling units.\n"
        )
    if path.endswith("service-nologin.php"):
        return httpx.Response(
            200, json=[{"error": False, "data": {"sitename": "IU", "enablemobilewebservice": 1}}]
        )
    return httpx.Response(404)


@pytest.fixture
def client() -> MoodleClient:
    http = httpx.Client(transport=httpx.MockTransport(_handler))
    return MoodleClient(SITE, TOKEN, http=http)


def test_client_errors_are_raised(client):
    with pytest.raises(MoodleError) as info:
        client.call("no_such_function")
    assert info.value.errorcode == "invalidfunction"


def test_public_config(client):
    config = MoodleClient.public_config(SITE, http=client.http)
    assert config["enablemobilewebservice"] == 1


def test_sync_collects_and_indexes(settings, store, client):
    ingestor = Ingestor(settings, store)
    sync = MoodleSync(settings, client, ingestor)
    plan = sync.collect()
    sources = {d.source for d in plan.documents}
    assert "moodle:42/100/book.md" in sources
    assert "moodle:42/101/page" in sources
    assert "moodle:42/200/assignment" in sources
    assert "moodle:42/forum-5/discussion-77" in sources
    assert "moodle:42/103/url" in sources
    assert "moodle:42/section-1" in sources
    assert any("video.mp4" in s for s in plan.skipped)
    assert any("Self test" in s for s in plan.skipped)
    assert not any("forum-6" in s for s in sources)  # only news forums by default

    report, plan = sync.run()
    assert report.failed == []
    assert report.added == len(plan.documents)
    cached = settings.moodle_cache_dir / "42" / "100_book.md"
    assert cached.exists()

    hits = store.search("dropout regularises", k=1)
    assert hits[0][0].metadata["module"] == "Course book"
    assert hits[0][0].metadata["origin"] == "moodle"
    hits = store.search("exam december", k=1)
    assert hits[0][0].metadata["module_type"] == "forum"

    # second run: everything unchanged
    report, _ = sync.run()
    assert report.added == 0 and report.skipped > 0

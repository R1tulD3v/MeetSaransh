"""Upload content validation, rate limiting, and response headers."""

from __future__ import annotations

import time

import pytest

from app import security
from tests.conftest import mp3_bytes, wav_bytes


# ------------------------------------------------------------------ magic-byte sniffing
@pytest.mark.parametrize(
    ("head", "expected"),
    [
        (b"ID3\x04\x00\x00\x00\x00\x00\x00\x00\x00", "mp3"),
        (b"\xff\xfb\x90\x64" + b"\x00" * 8, "mp3"),  # raw MPEG frame sync, no ID3 tag
        (b"RIFF\x24\x00\x00\x00WAVEfmt ", "wav"),
        (b"\x00\x00\x00\x20ftypM4A ", "mp4"),
        (b"\x1a\x45\xdf\xa3" + b"\x00" * 8, "webm"),
        (b"OggS\x00\x02\x00\x00\x00\x00\x00\x00", "ogg"),
        (b"fLaC\x00\x00\x00\x22\x00\x00\x00\x00", "flac"),
    ],
)
def test_recognises_real_audio_containers(head, expected):
    assert security.sniff_audio_format(head) == expected


@pytest.mark.parametrize(
    "head",
    [
        b"%PDF-1.7\n%\xe2\xe3\xcf\xd3",  # a PDF renamed to .mp3
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\x00",  # a PNG
        b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00",  # a Windows executable
        b"#!/bin/sh\nrm -rf /\n",  # a shell script
        b"PK\x03\x04\x14\x00\x00\x00\x08\x00\x00\x00",  # a zip
    ],
)
def test_rejects_non_audio_content(head):
    assert security.sniff_audio_format(head) is None


def test_truncated_input_is_not_guessed_at():
    assert security.sniff_audio_format(b"ID3") is None  # too short to be sure


def test_riff_that_is_not_wave_is_rejected():
    """RIFF is a container family; only the WAVE payload is audio we accept."""
    assert security.sniff_audio_format(b"RIFF\x24\x00\x00\x00AVI LIST") is None


# ------------------------------------------------------------------- extension matching
def test_valid_audio_matching_its_extension_passes():
    assert security.validate_audio_content(mp3_bytes()[:64], ".mp3") == "mp3"
    assert security.validate_audio_content(wav_bytes()[:64], ".wav") == "wav"


def test_a_renamed_executable_is_rejected_by_content():
    """The whole point: an extension check alone would have let this through."""
    with pytest.raises(security.UnsupportedAudioError, match="does not look like audio"):
        security.validate_audio_content(b"MZ\x90\x00" + b"\x00" * 60, ".mp3")


def test_real_audio_with_a_mismatched_extension_is_rejected():
    with pytest.raises(security.UnsupportedAudioError, match="claims"):
        security.validate_audio_content(wav_bytes()[:64], ".mp3")


def test_an_unlisted_extension_can_never_match():
    with pytest.raises(security.UnsupportedAudioError):
        security.validate_audio_content(mp3_bytes()[:64], ".exe")


# ------------------------------------------------------------------------ rate limiting
def test_requests_are_allowed_up_to_the_limit_then_rejected():
    limiter = security.RateLimiter()
    for i in range(3):
        allowed, remaining, _ = limiter.check("client", limit=3, window=60)
        assert allowed is True
        assert remaining == 2 - i

    allowed, remaining, retry_after = limiter.check("client", limit=3, window=60)
    assert allowed is False
    assert remaining == 0
    assert 0 < retry_after <= 60


def test_limits_are_tracked_per_key():
    limiter = security.RateLimiter()
    assert limiter.check("alice", 1, 60)[0] is True
    assert limiter.check("alice", 1, 60)[0] is False
    assert limiter.check("bob", 1, 60)[0] is True  # bob is unaffected by alice


def test_the_window_slides_so_capacity_returns():
    limiter = security.RateLimiter()
    assert limiter.check("k", 1, window=0.05)[0] is True
    assert limiter.check("k", 1, window=0.05)[0] is False
    time.sleep(0.06)
    assert limiter.check("k", 1, window=0.05)[0] is True


def test_a_sliding_window_prevents_the_boundary_burst():
    """A fixed window would allow 2x the limit across a boundary; this must not.

    Costly here means a real charge from the ASR provider, so the distinction is a
    money bug rather than a purity argument.
    """
    limiter = security.RateLimiter()
    window = 0.30
    for _ in range(2):
        assert limiter.check("k", 2, window)[0] is True
    time.sleep(window * 0.6)  # past the midpoint, before the first hit expires
    assert limiter.check("k", 2, window)[0] is False


def test_prune_drops_stale_buckets_but_keeps_active_ones():
    limiter = security.RateLimiter()
    limiter.check("stale", 5, 60)
    time.sleep(0.05)
    limiter.check("fresh", 5, 60)

    assert limiter.prune(older_than=0.04) == 1
    assert limiter.check("stale", 1, 60)[0] is True  # bucket was forgotten
    assert "fresh" in limiter._hits


def test_reset_clears_everything():
    limiter = security.RateLimiter()
    limiter.check("k", 1, 60)
    limiter.reset()
    assert limiter.check("k", 1, 60)[0] is True


# --------------------------------------------------------------------- endpoint budgets
def test_the_transcription_endpoint_gets_the_tightest_budget():
    upload = security.limit_for_path("POST", "/api/v1/meetings")
    chat = security.limit_for_path("POST", "/api/v1/chat")
    default = security.limit_for_path("GET", "/api/v1/meetings")
    assert upload < chat < default


def test_reading_a_meeting_is_not_charged_the_upload_budget():
    """Only POST /meetings runs ASR; GET on the same path must not share its cap."""
    assert security.limit_for_path("GET", "/api/v1/meetings") != security.limit_for_path(
        "POST", "/api/v1/meetings"
    )


# ------------------------------------------------------------------- response headers
def test_the_csp_is_strict_enough_to_be_worth_having():
    csp = security.security_headers(is_production=False)["Content-Security-Policy"]
    assert "default-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "object-src 'none'" in csp
    # The frontend has no inline script or style, so no unsafe escape hatch is needed.
    assert "unsafe-inline" not in csp
    assert "unsafe-eval" not in csp


def test_hsts_is_production_only():
    """HSTS over a local http:// dev server would pin the browser to a broken scheme."""
    assert "Strict-Transport-Security" not in security.security_headers(is_production=False)
    assert "Strict-Transport-Security" in security.security_headers(is_production=True)

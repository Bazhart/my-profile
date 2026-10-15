#!/usr/bin/env python3
"""
run_pipeline.py - recreate the "diner selfie" reel with two friends (Pierre + Basil) on fal.ai.

Steps (subcommands):
  refs      1. Identity-preserving photo edit: one 16:9 waist-up portrait per friend in a dark navy
               two-piece suit + white shirt (2-3 variants each). STOPS so a human picks the best file.
  replace   2. Character replacement on the template reel.
               Strategy "wan-isolate" (default): fal Wan 2.2 Animate /replace, run TWICE (once per friend).
               That endpoint has NO person selector (it keeps the largest person, frame by frame), so the
               script greys out the other people before each pass (--isolate head: in Pierre's pass Basil's whole
               box is hidden and only Pierre's head box is given back; in Basil's pass Pierre's box AND head are
               hidden), runs a free pre-flight that refuses to pay if the other man's head is visible in any frame
               (optional YOLO simulation of Wan's choice with --preflight-yolo), gives Basil's pass a static zoom
               crop, then composites both results back onto the original with a difference matte limited to each
               person's box and to the area that pass actually saw.
               Strategy "recast": minimax/h3-max/recast one-pass alternative (both people, 2 shots).
  lipsync   3. Optional lip repair with fal sync-lipsync v3, targeting each speaker's face explicitly.
  finish    4. Local ffmpeg: retime to the template's exact frame count/fps/size, put the ORIGINAL template
               audio back (stream copy), export 16:9 H.264/AAC mp4 and an optional 9:16 crop.
  all       refs -> (stop unless --pierre-image/--basil-image given) -> replace -> [lipsync] -> finish
  tracks-preview   Local check of the person tracks (coloured boxes over the template).

Requirements: python3, ffmpeg/ffprobe, `pip install fal-client==1.0.3`. No GPU, nothing heavy runs locally.
Auth: the fal key is read from the environment variable FAL_KEY only (by fal-client). It is never printed.
Every request/response JSON and every downloaded file is stored in --out. Cached files and paid calls are named
after a hash of their inputs (photos by content + parameters), so a rerun with the same inputs resumes/skips and a
rerun with different inputs (another photo, margin, threshold...) makes new files instead of reusing old ones.
A submitted request's id is kept in <call>.pending.json until the request has definitely ended: a crash, timeout
(--max-wait) or transient HTTP error never leads to a second paid submission.

Sources: the endpoint ids and field names below were checked against fal's generated client types
(npm @fal-ai/client 1.11.0-alpha.5 src/types/endpoints.d.ts == github.com/fal-ai/fal-js
libs/client/src/types/endpoints.ts @959d1647, 2026-10-13) unless marked "# UNVERIFIED".
Prices come from fal model pages as quoted by web-search summaries (fal.ai could not be fetched): re-check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRACKS = SCRIPT_DIR / "reel1_tracks.json"

# ----------------------------------------------------------------------------------------------
# Endpoint ids and field names
# ----------------------------------------------------------------------------------------------
# Image edit (step "refs"). Chain order = fallback order. A 422 (e.g. content_policy_violation)
# or a result with an empty images[] moves on to the next model.
#
# https://fal.ai/models/fal-ai/nano-banana-pro/edit  (types: NanoBananaProEditInput)
#   fields: prompt, image_urls[], aspect_ratio(auto|21:9|16:9|...), resolution(1K|2K|4K), num_images,
#           output_format(jpeg|png|webp), seed, safety_tolerance("1".."6" string), sync_mode
#   num_images max 4 and max 14 refs come from the fal page only  # UNVERIFIED
# https://fal.ai/models/bytedance/seedream/v5/pro/edit  (types: SeedreamV5ProEditInput)
#   fields: prompt, image_urls[] (<=10, last 10 used), image_size(enum|{width,height}, 1024^2..2048^2 px),
#           num_images, output_format(jpeg|png), enable_safety_checker. NO seed.
# https://fal.ai/models/fal-ai/flux-2-pro/edit  (types: Flux2ProEditInput)
#   fields: prompt, image_urls[], image_size(enum|{w,h}), output_format, seed, safety_tolerance("1".."5")
#   NO num_images -> one call per variant.
# https://fal.ai/models/fal-ai/nano-banana-2/edit  (types: NanoBanana2EditInput) - cheaper Google sibling.
REF_MODELS = {
    "nano-banana-pro": {
        "endpoint": "fal-ai/nano-banana-pro/edit",
        "max_refs": 14,          # UNVERIFIED (fal page only)
        "max_num_images": 4,     # UNVERIFIED (fal page only)
        "seed": True,
        "usd_per_image": 0.15,   # 1K/2K, fal page via search
    },
    "seedream-v5-pro": {
        "endpoint": "bytedance/seedream/v5/pro/edit",
        "max_refs": 10,
        "max_num_images": 4,     # UNVERIFIED
        "seed": False,
        "usd_per_image": 0.0675,  # "tentative" on fal page, <=1536x1536 tier
    },
    "flux-2-pro": {
        "endpoint": "fal-ai/flux-2-pro/edit",
        "max_refs": 9,           # UNVERIFIED (fal page only)
        "max_num_images": 1,
        "seed": True,
        "usd_per_image": 0.05,   # ~$0.03 first MP + $0.015/extra MP incl. inputs (fal page via search)
    },
    "nano-banana-2": {
        "endpoint": "fal-ai/nano-banana-2/edit",
        "max_refs": 14,          # UNVERIFIED
        "max_num_images": 4,     # UNVERIFIED
        "seed": True,
        "usd_per_image": 0.12,   # 2K, fal page via search
    },
}
DEFAULT_REF_CHAIN = "nano-banana-pro,seedream-v5-pro,flux-2-pro"

# Character replacement (step "replace").
# https://fal.ai/models/fal-ai/wan/v2.2-14b/animate/replace  (types: WanV2214bAnimateReplaceInput)
#   fields: video_url, image_url, resolution(480p|580p|720p), num_inference_steps(20), guidance_scale(1),
#           shift(5, 1..10), seed, use_turbo, video_quality(low|medium|high|maximum),
#           video_write_mode(fast|balanced|small), return_frames_zip, enable_safety_checker,
#           enable_output_safety_checker.  Output: {video:{url}, prompt, seed, frames_zip?}
#   No person selector. Person choice = largest box per frame in the open-source preprocessing
#   (Wan2.2 pose2d.py select_type='max'); that fal runs the same code is  # UNVERIFIED
#   Output fps/frame count and audio retention on fal  # UNVERIFIED (the script re-times and re-muxes).
WAN_REPLACE = "fal-ai/wan/v2.2-14b/animate/replace"
WAN_USD_PER_BILLED_SECOND = {"480p": 0.04, "580p": 0.06, "720p": 0.08}  # billed as frames/16 (fal page via search)

# https://fal.ai/models/minimax/h3-max/recast  (fal API page + third-party zod schema)
#   fields: video_url (5-30 s, each shot <=15 s), reference_image_urls (1-4), resolution(768P|1080P),
#           prompt (<=2000), seed.  Output: {video:{url}, seed}
RECAST = "minimax/h3-max/recast"
RECAST_USD_PER_SECOND = {"768P": 0.30, "1080P": 0.45}

# Lip-sync repair (step "lipsync").
# https://fal.ai/models/fal-ai/sync-lipsync/v3  (types: SyncLipsyncV3Input)
#   fields: video_url, audio_url, sync_mode(cut_off|loop|bounce|silence|remap),
#           options.active_speaker_detection{auto_detect, coordinates[x,y], frame_number,
#           bounding_boxes, bounding_boxes_url, face_image}
#   The options.active_speaker_detection fields are in fal's generated types but not on the public page.
#   Coordinate units (pixels of the uploaded video)  # UNVERIFIED
#   auto_detect=False being the switch that makes the model use coordinates/frame_number  # UNVERIFIED
#   (--sync-auto-detect omit sends the coordinates without the auto_detect key).
#   Output resolution / fps of sync v3 on fal  # UNVERIFIED (the script re-conforms before pass B).
SYNC_V3 = "fal-ai/sync-lipsync/v3"
SYNC_V3_USD_PER_MIN = 8.0

UPLOAD_EXPIRES = "7d"   # fal_client.StorageSettings(expires_in=...), values verified in fal_client 1.0.3 source
UPLOAD_REUSE_MAX_AGE_S = 6 * 24 * 3600   # re-upload cached files older than this (uploads expire after 7 d)
DEFAULT_MAX_WAIT_S = 3600                # per queued call; pending.json is kept when it is reached

# ----------------------------------------------------------------------------------------------
# Prompts (step "refs")
# ----------------------------------------------------------------------------------------------
PEOPLE_DEFAULT_DESC = {
    "pierre": "same blond hair, slicked back and tied in a short low ponytail",
    "basil": "short tousled brown hair and moustache",
}
PERSON_EXTRA = {
    # Pierre's only frontal photo is a pout shot from a low angle with hard side light.
    "pierre": ("The first reference photo is the identity anchor but he is pouting there and the camera is below "
               "him: do NOT copy the pout, the low angle or the hard side light. Remove any sunglasses; eyes open "
               "and looking into the lens."),
    "basil": ("If a reference photo contains other people, use only this man. No cap, no hat, no sunglasses; "
              "eyes open and looking into the lens."),
}


def ref_prompt(person: str, desc: str) -> str:
    return (
        "Photorealistic portrait photo of the man shown in the reference photos (all reference photos show the "
        "same man). Keep exactly the same face and identity: same face shape, eyes, nose, mouth, ears, skin tone, "
        f"age and hairline; {desc}. {PERSON_EXTRA[person]} "
        "Neutral relaxed expression, lips gently closed, no pout, no grin. "
        "Camera at eye level, straight-on, not from below, 50mm lens. Soft even frontal daylight, no harsh shadows. "
        "Upper body, waist-up, facing the camera, both shoulders visible, centered in a 16:9 landscape frame with "
        "headroom above the head, alone in the frame. "
        "He wears a tailored dark navy blue two-piece Italian suit in fine wool, slim fit, notch lapels, "
        "with a crisp white dress shirt, top button open, no tie. "
        "Plain light-grey seamless studio background. Natural skin texture, sharp focus, no retouching, "
        "no text, no watermark."
    )


# Reference order sent to recast = [Basil, Pierre]. Recast's documented default assigns photos to the main
# people from LEFT to RIGHT; Basil (man across, centre x ~0.5) is left of Pierre (selfie holder, x ~0.8), so the
# default mapping and this prompt agree. Whether the prompt can override the mapping is  # UNVERIFIED
RECAST_PROMPT = (
    "Replace the two main men and keep everything else identical (camera motion, cuts, diner background, "
    "lighting, timing, gestures and mouth movements). Reference Image 1 is the man in the dark tracksuit seated "
    "across the counter (the man on the left of the two). Reference Image 2 is the man in the foreground on the "
    "right who holds the phone (selfie point of view). Both now wear a tailored dark navy blue two-piece Italian suit "
    "with a crisp white dress shirt, top button open, no tie. The man in the blue suit further back and anyone "
    "else stay exactly as they are. Photorealistic."
)

# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------


class PipelineError(Exception):
    pass


class Refusal(Exception):
    """Model refused or returned nothing usable; caller may try a fallback."""


class ModelError(PipelineError):
    """The request definitely ended with an HTTP error (nothing pending to resume); a fallback may be tried."""

    def __init__(self, msg: str, status_code: int | None = None):
        super().__init__(msg)
        self.status_code = status_code


def log(msg: str) -> None:
    print(msg, flush=True)


def die(msg: str) -> None:
    raise PipelineError(msg)


def need_tool(name: str) -> None:
    if shutil.which(name) is None:
        die(f"'{name}' not found in PATH (install ffmpeg).")


def run(cmd: list[str], *, quiet: bool = True, stdin=None) -> subprocess.CompletedProcess:
    full = [cmd[0], "-hide_banner", "-loglevel", "error", "-y"] + cmd[1:] if cmd[0] == "ffmpeg" else cmd
    r = subprocess.run(full, capture_output=True, text=True, input=stdin)
    if r.returncode != 0:
        die(f"command failed ({r.returncode}): {' '.join(full)[:600]}\n{r.stderr[-2000:]}")
    return r


def ffprobe(path: Path) -> dict:
    r = run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-count_frames", "-of", "json", str(path)])
    d = json.loads(r.stdout)
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in d.get("streams", []) if s.get("codec_type") == "audio"), None)
    info = {"duration": float(d.get("format", {}).get("duration", 0) or 0), "has_audio": a is not None}
    if v:
        num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1").split("/")
        fps = float(num) / float(den) if float(den) else 0.0
        frames = int(v.get("nb_read_frames") or v.get("nb_frames") or round(fps * info["duration"]))
        info.update(width=int(v["width"]), height=int(v["height"]), fps=fps, frames=frames,
                    vduration=float(v.get("duration") or info["duration"]))
    if a:
        info.update(audio_codec=a.get("codec_name"), audio_duration=float(a.get("duration") or 0),
                    sample_rate=int(a.get("sample_rate") or 0), channels=int(a.get("channels") or 0))
    return info


def probe_image(path: Path) -> tuple[int, int]:
    r = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
             "-of", "csv=p=0", str(path)])
    w, h = r.stdout.strip().split(",")[:2]
    return int(w), int(h)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_SHA_MEMO: dict[tuple, str] = {}


def file_sha(path: Path) -> str:
    """sha256 of a file, memoised on (path, size, mtime)."""
    st = Path(path).stat()
    k = (str(Path(path).resolve()), st.st_size, st.st_mtime_ns)
    if k not in _SHA_MEMO:
        _SHA_MEMO[k] = sha256(Path(path))
    return _SHA_MEMO[k]


def fingerprint(*parts) -> str:
    """8-hex hash of the inputs of a step. Path objects are hashed by content, everything else by value.
    Used in cached file/call names so that changing an input (image, parameter, template) never silently
    reuses an old result, while switching back to earlier inputs reuses what was already paid for."""
    def norm(v):
        if isinstance(v, Path):
            return {"file_sha256": file_sha(v)} if v.exists() else {"missing": str(v)}
        if isinstance(v, dict):
            return {k: norm(x) for k, x in sorted(v.items())}
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        return v
    return hashlib.sha256(json.dumps(norm(list(parts)), sort_keys=True).encode()).hexdigest()[:8]


def host_hint(e: Exception, what: str) -> None:
    """Turn a network error into a message naming the host to allowlist (never prints headers/keys)."""
    host = "?"
    try:
        host = e.request.url.host  # httpx errors carry the request
    except Exception:  # noqa: BLE001
        pass
    die(f"{what}: cannot reach host '{host}' ({type(e).__name__}: {str(e)[:200]}). If the network uses an "
        f"allowlist, add '{host}' to it, then rerun the same command (it resumes).")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def exists(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def even(x: float) -> int:
    return max(2, int(round(x / 2.0)) * 2)


def parse_path_crop(spec: str) -> tuple[Path, tuple[int, int, int, int] | None]:
    """'photo.png' or 'photo.png@x,y,w,h' (crop in source pixels)."""
    if "@" in spec:
        p, c = spec.rsplit("@", 1)
        try:
            x, y, w, h = (int(v) for v in c.split(","))
        except ValueError:
            die(f"bad crop spec in '{spec}', expected path@x,y,w,h")
        return Path(p), (x, y, w, h)
    return Path(spec), None


# ----------------------------------------------------------------------------------------------
# fal access layer (upload / queue call / download), with persistence + resume + dry-run
# ----------------------------------------------------------------------------------------------


class Fal:
    def __init__(self, out: Path, dry_run: bool, yes: bool, max_wait: float = DEFAULT_MAX_WAIT_S):
        self.out, self.dry, self.yes, self.max_wait = out, dry_run, yes, max_wait
        self.cache_path = out / "uploads.json"
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.planned_usd = 0.0
        self._client = None

    # -- auth: fal-client reads FAL_KEY from the environment itself; we only check presence. --
    def client(self):
        if self._client is None:
            if not os.environ.get("FAL_KEY"):
                die("FAL_KEY is not set. Export your fal.ai API key as FAL_KEY in the environment "
                    "(do not put it in files or on the command line), or use --dry-run.")
            try:
                import fal_client  # noqa: WPS433
            except ImportError:
                die("fal-client is not installed: pip install 'fal-client==1.0.3'")
            self._client = fal_client
        return self._client

    def note_host(self, url: str) -> None:
        host = urllib.parse.urlparse(url).hostname or "?"
        hp = self.out / "hosts_seen.txt"
        seen = set(hp.read_text().split()) if hp.exists() else set()
        if host not in seen:
            hp.parent.mkdir(parents=True, exist_ok=True)
            with open(hp, "a") as f:
                f.write(host + "\n")

    @staticmethod
    def _net_errors():
        import httpx  # dependency of fal-client
        return (httpx.TransportError,)

    def upload(self, path: Path) -> str:
        path = Path(path)
        if self.dry and not path.exists():   # output of an earlier planned (not executed) call
            return f"<upload:{path.name}>"
        key = f"{file_sha(path)}:{path.stat().st_size}"
        hit = self.cache.get(key)
        if hit:
            age = time.time() - float(hit.get("uploaded_epoch") or 0)
            if age < UPLOAD_REUSE_MAX_AGE_S:
                return hit["url"]
            log(f"  cached upload of {path.name} is {age/86400:.1f} days old (uploads expire after "
                f"{UPLOAD_EXPIRES}): uploading again")
        if self.dry:
            return f"<upload:{path.name}>"
        fc = self.client()
        log(f"  upload {path.name} ({path.stat().st_size/1e6:.1f} MB)")
        try:
            url = fc.upload_file(path, lifecycle=fc.StorageSettings(expires_in=UPLOAD_EXPIRES))
        except self._net_errors() as e:
            host_hint(e, f"upload of {path.name}")
        except fc.FalClientHTTPError as e:
            die(f"upload of {path.name} failed: HTTP {e.status_code}: {str(e)[:400]}")
        self.note_host(url)
        self.cache[key] = {"url": url, "file": path.name, "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                           "uploaded_epoch": time.time()}
        write_json(self.cache_path, self.cache)
        return url

    def confirm(self, what: str, usd: float) -> None:
        self.planned_usd += usd
        if self.dry or self.yes:
            return
        if not sys.stdin.isatty():
            die(f"paid call needs confirmation ({what}, ~${usd:.2f}); rerun with --yes")
        ans = input(f"  PAID CALL {what}: estimated ~${usd:.2f} (estimate, check fal pricing). Continue? [y/N] ")
        if ans.strip().lower() not in ("y", "yes", "o", "oui"):
            die("cancelled by user")

    def call(self, calls_dir: Path, name: str, endpoint: str, arguments: dict, usd: float) -> dict | None:
        """Queue call with persistence. Returns the response JSON, or None in dry-run.
        `name` must encode the call's inputs (see fingerprint()) so that a cached response is only reused for
        the same inputs. The pending file (request_id) is deleted only once the request has definitely ended,
        so a crash, timeout or transient HTTP error while polling never leads to a second paid submission."""
        req_p = calls_dir / f"{name}.request.json"
        resp_p = calls_dir / f"{name}.response.json"
        pend_p = calls_dir / f"{name}.pending.json"
        if resp_p.exists():
            log(f"  [skip] {name}: response already saved ({resp_p.name})")
            return json.loads(resp_p.read_text())
        if self.dry:
            self.planned_usd += usd
            log(f"  [dry-run] would call {endpoint}  (~${usd:.2f})  [{name}]")
            log("    arguments = " + json.dumps(arguments, indent=2, ensure_ascii=False).replace("\n", "\n    "))
            return None
        fc = self.client()
        net = self._net_errors()
        if pend_p.exists():
            request_id = json.loads(pend_p.read_text())["request_id"]
            log(f"  resume {name}: request_id {request_id}")
        else:
            self.confirm(f"{endpoint} [{name}]", usd)
            write_json(req_p, {"endpoint": endpoint, "arguments": arguments,
                               "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            try:
                handle = fc.submit(endpoint, arguments)
            except net as e:
                host_hint(e, f"submit {name}")
            except fc.FalClientHTTPError as e:
                self._fail(calls_dir, name, e, None)   # not queued: nothing to resume
            request_id = handle.request_id
            write_json(pend_p, {"endpoint": endpoint, "request_id": request_id,
                                "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            log(f"  submitted {endpoint} -> request_id {request_id}")

        def keep_pending(why: str):
            write_json(calls_dir / f"{name}.poll_error.json", {"request_id": request_id, "error": why[:2000],
                                                                "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            die(f"{name}: {why[:400]}\n  The request may still be running on fal: {pend_p} is KEPT, rerun the same "
                f"command to resume request_id {request_id} without paying again (delete that file only if you "
                f"really want a new paid submission).")

        t0, last = time.time(), ""
        while True:
            if time.time() - t0 > self.max_wait:
                keep_pending(f"still not completed after --max-wait {self.max_wait:.0f} s (last status: {last})")
            try:
                st = fc.status(endpoint, request_id, with_logs=True)
            except net as e:
                try:
                    host_hint(e, f"status of {name}")
                except PipelineError as pe:
                    keep_pending(str(pe))
            except fc.FalClientHTTPError as e:
                keep_pending(f"status poll failed: HTTP {e.status_code} {e.error_type}: {str(e)[:300]}")
            label = type(st).__name__
            if isinstance(st, fc.Queued):
                label += f" (position {st.position})"
            if isinstance(st, fc.InProgress) and st.logs:
                msg = str(st.logs[-1].get("message", ""))[:120]
                label += f" {msg}"
            if label != last:
                log(f"    [{int(time.time()-t0):4d}s] {label}")
                last = label
            if isinstance(st, fc.Completed):
                if getattr(st, "error", None):  # completed with an error (e.g. content policy): request is over
                    write_json(calls_dir / f"{name}.error.json",
                               {"request_id": request_id, "status": "COMPLETED", "error": st.error,
                                "error_type": st.error_type})
                    pend_p.unlink(missing_ok=True)
                    raise Refusal(f"{name}: {st.error_type}: {str(st.error)[:400]}")
                break
            time.sleep(5)
        try:
            result = fc.result(endpoint, request_id)
        except net as e:
            try:
                host_hint(e, f"result of {name}")
            except PipelineError as pe:
                keep_pending(str(pe))
        except fc.FalClientHTTPError as e:
            if 400 <= e.status_code < 500 and e.status_code not in (401, 403, 404, 408, 409, 429):
                # status said COMPLETED and the result itself is an error (e.g. 422): the request is over.
                pend_p.unlink(missing_ok=True)
                self._fail(calls_dir, name, e, request_id)
            keep_pending(f"result fetch failed: HTTP {e.status_code} {e.error_type}: {str(e)[:300]}")
        write_json(resp_p, result)
        pend_p.unlink(missing_ok=True)
        return result

    def _fail(self, calls_dir: Path, name: str, e, request_id: str | None) -> None:
        body = None
        try:
            body = e.response.json()
        except Exception:  # noqa: BLE001
            body = str(e)[:2000]
        write_json(calls_dir / f"{name}.error.json",
                   {"request_id": request_id, "status_code": e.status_code, "error_type": e.error_type,
                    "body": body})
        if e.status_code == 422:
            kind = "content_policy_violation" if "content_policy" in json.dumps(body) else "validation/422"
            raise Refusal(f"{name}: HTTP 422 ({kind}): {str(body)[:400]}")
        if e.status_code in (401, 403):
            die(f"{name}: HTTP {e.status_code} (authentication/permission: check FAL_KEY and account): "
                f"{str(body)[:300]}")
        raise ModelError(f"{name}: HTTP {e.status_code}: {str(body)[:600]}", e.status_code)

    def download(self, url: str, dest: Path) -> Path:
        if exists(dest):
            return dest
        if url.startswith("data:"):
            import base64
            dest.write_bytes(base64.b64decode(url.split(",", 1)[1]))
            return dest
        self.note_host(url)
        import httpx  # dependency of fal-client
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=httpx.Timeout(300, connect=30)) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
        except httpx.TransportError as e:
            host_hint(e, f"download of {dest.name}")
        tmp.replace(dest)
        log(f"  downloaded {dest}")
        return dest


# ----------------------------------------------------------------------------------------------
# Person tracks (keyframed boxes) and mask rendering (stdlib + ffmpeg, no numpy)
# ----------------------------------------------------------------------------------------------


class Tracks:
    """Keyframes [frame, x1, y1, x2, y2] in template pixels, linearly interpolated."""

    def __init__(self, path: Path):
        d = json.loads(Path(path).read_text())
        self.path = Path(path)
        self.W, self.H, self.fps, self.n = d["W"], d["H"], d["fps"], d["n_frames"]
        self.people = dict(d["tracks"])
        self.others = dict(d.get("others", {}))
        self.faces = d.get("faces", {})
        self.heads = dict(d.get("heads", {}))
        for k in ("pierre", "basil"):
            if not self.people.get(k):
                die(f"tracks file {path} has no '{k}' track")

    def has_head(self, name: str) -> bool:
        return bool(self.heads.get(name))

    def head(self, name: str, f: float):
        """Head box at frame f, or None when the head is out of frame (zero-size keyframes) or not tracked."""
        kf = self.heads.get(name)
        if not kf:
            return None
        b = self._interp(kf, f)
        return b if (b[2] - b[0]) > 2 and (b[3] - b[1]) > 2 else None

    def all_names(self):
        return list(self.people) + list(self.others)

    def kf(self, name):
        return self.people.get(name) or self.others.get(name)

    def box(self, name: str, f: float):
        return self._interp(self.kf(name), f)

    @staticmethod
    def _interp(kf, f: float):
        if f <= kf[0][0]:
            return kf[0][1:]
        if f >= kf[-1][0]:
            return kf[-1][1:]
        for a, b in zip(kf, kf[1:]):
            if a[0] <= f <= b[0]:
                u = (f - a[0]) / (b[0] - a[0]) if b[0] != a[0] else 0
                return [a[i] + (b[i] - a[i]) * u for i in range(1, 5)]
        return kf[-1][1:]

    def check_template(self, info: dict) -> None:
        if (info["width"], info["height"]) != (self.W, self.H) or abs(info["frames"] - self.n) > 2:
            log(f"  WARNING: tracks {self.path.name} were made for {self.W}x{self.H}/{self.n} frames, template is "
                f"{info['width']}x{info['height']}/{info['frames']} frames. Make a matching tracks file.")


def scaled_box(b, sx, sy, W, H, margin=0.0):
    x1, y1, x2, y2 = b
    mw, mh = (x2 - x1) * margin, (y2 - y1) * margin
    return (max(0, int((x1 - mw) * sx)), max(0, int((y1 - mh) * sy)),
            min(W, int((x2 + mw) * sx)), min(H, int((y2 + mh) * sy)))


def render_mask(dest: Path, W: int, H: int, n: int, fps: float, frame_ops) -> Path:
    """frame_ops(i) -> list of (x1, y1, x2, y2, value 0..255) painted in order. Writes lossless FFV1 gray."""
    if exists(dest):
        return dest
    tmp = dest.with_name(dest.stem + ".part" + dest.suffix)
    p = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-",
                          "-c:v", "ffv1", "-pix_fmt", "gray", str(tmp)], stdin=subprocess.PIPE)
    rows = {v: bytes([v]) * W for v in (0, 255)}
    for i in range(n):
        buf = bytearray(W * H)
        for x1, y1, x2, y2, v in frame_ops(i):
            if x2 <= x1 or y2 <= y1:
                continue
            row = rows.setdefault(v, bytes([v]) * W)[: x2 - x1]
            for y in range(y1, y2):
                o = y * W
                buf[o + x1: o + x2] = row
        p.stdin.write(buf)
    p.stdin.close()
    if p.wait() != 0:
        die(f"mask rendering failed for {dest}")
    tmp.replace(dest)
    return dest


def piecewise_expr(points: list[tuple[float, float]]) -> str:
    """ffmpeg expression of t: linear interpolation through (t, value) points, held at both ends."""
    pts = sorted(points)
    parts = [f"lt(t,{pts[0][0]:.4f})*{pts[0][1]:.2f}"]
    for (ta, va), (tb, vb) in zip(pts, pts[1:]):
        if tb <= ta:
            continue
        k = (vb - va) / (tb - ta)
        parts.append(f"gte(t,{ta:.4f})*lt(t,{tb:.4f})*({va:.2f}+(t-{ta:.4f})*{k:.4f})")
    parts.append(f"gte(t,{pts[-1][0]:.4f})*{pts[-1][1]:.2f}")
    return "+".join(parts)


# ----------------------------------------------------------------------------------------------
# Local video helpers
# ----------------------------------------------------------------------------------------------


def conform(src: Path, dst: Path, n: int, fps: float, W: int, H: int, ref_duration: float) -> Path:
    """Re-time any generated clip to exactly n frames at fps and WxH (video only)."""
    if exists(dst):
        return dst
    info = ffprobe(src)
    if info.get("frames") == n:
        setpts = f"setpts=N/({fps}*TB)"            # same frame count: map frame i -> i/fps
        mode = "frame-for-frame"
    elif abs(info["vduration"] - ref_duration) <= 0.25:
        setpts = "setpts=PTS-STARTPTS"              # same duration, other fps: resample by time
        mode = "resample"
    else:
        ratio = ref_duration / max(info["vduration"], 1e-6)
        setpts = f"setpts=(PTS-STARTPTS)*{ratio:.6f}"  # different duration: stretch to template
        mode = f"stretch x{ratio:.4f}"
    log(f"  conform {src.name}: {info.get('frames')} f @ {info.get('fps', 0):.3f} fps, "
        f"{info.get('width')}x{info.get('height')} -> {n} f @ {fps} fps {W}x{H} ({mode})")
    vf = (f"{setpts},fps={fps},scale={W}:{H}:flags=lanczos,setsar=1,"
          f"tpad=stop_mode=clone:stop={n},trim=end_frame={n},setpts=N/({fps}*TB)")
    run(["ffmpeg", "-i", str(src), "-an", "-vf", vf, "-frames:v", str(n), "-c:v", "libx264", "-crf", "12",
         "-preset", "medium", "-pix_fmt", "yuv420p", "-r", f"{fps}", str(dst)])
    return dst


def fit_16x9_image(src: Path, dst: Path, W: int = 1280, H: int = 720) -> Path:
    """Character image for Wan: 16:9 without cropping the person (blurred fill), since fal center-crops."""
    if exists(dst):
        return dst
    fc = (f"[0:v]split[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=30[bg];"
          f"[b]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,format=rgb24")
    run(["ffmpeg", "-i", str(src), "-filter_complex", fc, "-frames:v", "1", str(dst)])
    return dst


# ----------------------------------------------------------------------------------------------
# Step 1: refs
# ----------------------------------------------------------------------------------------------


def prep_ref_inputs(person: str, specs: list[str], outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    res = []
    for i, spec in enumerate(specs, 1):
        src, crop = parse_path_crop(spec)
        if not src.is_file():
            die(f"reference photo not found: {src}")
        dst = outdir / f"{person}_ref{i}.png"
        vf = []
        if crop:
            x, y, w, h = crop
            vf.append(f"crop={w}:{h}:{x}:{y}")
        # 384..2048 px per side keeps every listed edit model happy; drop alpha.
        vf += ["scale='if(gt(iw,ih),min(2048,iw),-2)':'if(gt(iw,ih),-2,min(2048,ih))':flags=lanczos", "format=rgb24"]
        if not exists(dst):
            run(["ffmpeg", "-i", str(src), "-vf", ",".join(vf), "-frames:v", "1", str(dst)])
        w, h = probe_image(dst)
        if min(w, h) < 384:
            log(f"  WARNING: {dst.name} is only {w}x{h}; some models want >=384 px per side")
        res.append(dst)
    return res


def ref_arguments(model: str, prompt: str, urls: list[str], n: int, seed: int) -> list[dict]:
    """One dict per call (FLUX.2 has no num_images)."""
    spec = REF_MODELS[model]
    urls = urls[: spec["max_refs"]]
    if model in ("nano-banana-pro", "nano-banana-2"):
        out, left, k = [], n, 0
        while left > 0:
            m = min(left, spec["max_num_images"])
            out.append({"prompt": prompt, "image_urls": urls, "aspect_ratio": "16:9", "resolution": "2K",
                        "num_images": m, "output_format": "png", "seed": seed + k})
            left -= m
            k += 1
        return out
    if model == "seedream-v5-pro":
        # 1664x936 = 1.56 MP: inside 1024^2..2048^2 and inside the cheaper <=1536^2 price tier.
        out, left = [], n
        while left > 0:
            m = min(left, spec["max_num_images"])
            out.append({"prompt": prompt, "image_urls": urls, "image_size": {"width": 1664, "height": 936},
                        "num_images": m, "output_format": "png"})
            left -= m
        return out
    if model == "flux-2-pro":
        return [{"prompt": prompt, "image_urls": urls, "image_size": "landscape_16_9", "output_format": "png",
                 "seed": seed + i} for i in range(n)]
    die(f"unknown ref model {model}")


def contact_sheet(images: list[Path], dest: Path) -> None:
    """Side-by-side preview of the candidates (left to right = order of the printed file list)."""
    if not images:
        return
    ins = []
    for p in images:
        ins += ["-i", str(p)]
    cells = [f"[{i}:v]scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]"
             for i in range(len(images))]
    if len(images) == 1:
        graph = cells[0].replace("[v0]", "")
    else:
        graph = ";".join(cells) + ";" + "".join(f"[v{i}]" for i in range(len(images))) + f"hstack=inputs={len(images)}"
    run(["ffmpeg"] + ins + ["-filter_complex", graph, "-frames:v", "1", str(dest)])


def step_refs(a, fal: Fal) -> dict:
    out = a.out / "1_refs"
    results = {}
    who = [w.strip() for w in a.who.split(",") if w.strip()]
    chain = [m.strip() for m in a.ref_models.split(",") if m.strip()]
    for m in chain:
        if m not in REF_MODELS:
            die(f"unknown --ref-models entry '{m}'. Known: {', '.join(REF_MODELS)}")
    for person in who:
        specs = a.pierre_refs if person == "pierre" else a.basil_refs
        if not specs:
            die(f"--{person}-refs is required for step refs (who={a.who})")
        pdir = out / person
        desc = a.pierre_desc if person == "pierre" else a.basil_desc
        prompt = ref_prompt(person, desc)
        for spec_s in specs:
            src, _ = parse_path_crop(spec_s)
            if not src.is_file():
                die(f"reference photo not found: {src}")
        # Everything that changes the request goes into the fingerprint (photos by content, crops, prompt...).
        fp = fingerprint([(parse_path_crop(s)[0], parse_path_crop(s)[1]) for s in specs], prompt, a.variants,
                         a.seed, chain)
        done = pdir / "DONE.json"
        if done.exists():
            prev = json.loads(done.read_text())
            if prev.get("inputs_fp") == fp:
                log(f"[refs] {person}: already done for these inputs ({fp}) -> {pdir}")
                results[person] = prev
                continue
            log(f"[refs] {person}: inputs changed since the last run ({prev.get('inputs_fp')} -> {fp}); "
                f"making new candidates (old files are kept)")
        log(f"[refs] {person}: preparing {len(specs)} reference photo(s)  [inputs {fp}]")
        inputs = prep_ref_inputs(person, specs, pdir / f"inputs_{fp}")
        (pdir / f"prompt_{fp}.txt").write_text(prompt + "\n")
        urls = [fal.upload(p) for p in inputs]
        refused_p = pdir / f"refused_{fp}.json"
        refused = json.loads(refused_p.read_text()) if refused_p.exists() else {}
        saved, used = [], []
        for mi, model in enumerate(chain):
            spec = REF_MODELS[model]
            calls = ref_arguments(model, prompt, urls, a.variants, a.seed)
            if fal.dry and mi > 0:
                for args in calls:
                    short = {k: v for k, v in args.items() if k != "prompt"}
                    log(f"  [dry-run] fallback only if refused/failed: {spec['endpoint']} {json.dumps(short)}")
                continue
            if model in refused:
                log(f"  {model}: refused/failed earlier for these inputs ({refused[model][:160]}); skipped "
                    f"(delete {refused_p.name} to try it again)")
                continue
            got = []
            try:
                for ci, args in enumerate(calls):
                    nimg = args.get("num_images", 1)
                    r = fal.call(pdir / "calls", f"{model}_{fp}_{ci}", spec["endpoint"], args,
                                 nimg * spec["usd_per_image"])
                    if r is None:  # dry-run
                        continue
                    imgs = r.get("images") or []
                    if not imgs:
                        raise Refusal(f"{model}: empty images[] (description: {str(r.get('description'))[:200]})")
                    for j, im in enumerate(imgs):
                        ext = ".png" if "png" in (im.get("content_type") or "png") else ".jpg"
                        got.append(fal.download(im["url"], pdir / f"{person}_{model}_{fp}_c{ci}_{j+1}{ext}"))
            except (Refusal, ModelError) as e:
                # ModelError = definitive HTTP error other than 401/403 (a refusal may not always be a 422).
                log(f"  {model} refused/failed: {e}")
                refused[model] = str(e)[:500]
                write_json(refused_p, refused)
            if got:   # keep everything already paid for, even when a later call of the same model failed
                saved += got
                used.append(model)
            if saved and not fal.dry:
                if len(saved) < a.variants:
                    log(f"  only {len(saved)}/{a.variants} candidate(s); not paying another model for the rest")
                break
            if not fal.dry:
                log("  -> trying next model")
        if fal.dry:
            continue
        if not saved:
            die(f"[refs] {person}: every model in the chain refused or failed. See {pdir/'calls'}/*.error.json "
                f"and {refused_p}")
        contact_sheet(saved, pdir / f"contact_sheet_{fp}.jpg")
        shutil.copyfile(pdir / f"contact_sheet_{fp}.jpg", pdir / "contact_sheet.jpg")
        info = {"model": ",".join(used), "files": [str(p) for p in saved], "prompt": prompt, "inputs_fp": fp}
        write_json(done, info)
        results[person] = info
    if not fal.dry:
        log("\n=== STOP: pick the best image per person (identity first, no pout, suit OK) ===")
        for person, info in results.items():
            log(f"  {person}: {len(info['files'])} candidate(s), contact sheet {Path(info['files'][0]).parent}/contact_sheet.jpg")
            for f in info["files"]:
                log(f"     {f}")
        log("Then run:  run_pipeline.py replace --template <reel.mp4> --pierre-image <chosen> --basil-image <chosen> "
            f"--out {a.out}")
    return results


# ----------------------------------------------------------------------------------------------
# Step 2: replace
# ----------------------------------------------------------------------------------------------


def step_replace(a, fal: Fal) -> Path | None:
    out = a.out / "2_replace"
    out.mkdir(parents=True, exist_ok=True)
    final = out / "composite.mp4"
    for k in ("pierre_image", "basil_image"):
        p = getattr(a, k)
        if not p or not Path(p).is_file():
            die(f"--{k.replace('_', '-')} must point to the chosen step-1 image (got {p})")
    tpl = ffprobe(a.template)
    if not tpl.get("has_audio"):
        log("  WARNING: template has no audio stream")
    log(f"[replace] template {a.template.name}: {tpl['width']}x{tpl['height']} {tpl['fps']:.3f} fps, "
        f"{tpl['frames']} frames, {tpl['duration']:.3f} s; strategy={a.strategy}")
    if a.strategy == "recast":
        return replace_recast(a, fal, out, tpl, final)
    return replace_wan_isolate(a, fal, out, tpl, final)


def publish_composite(src: Path, a, final: Path, tpl: dict, fp: str) -> Path:
    """out/2_replace/composite.mp4 (+ composite.json naming the inputs it was built from)."""
    man = final.with_suffix(".json")
    if exists(final) and man.exists() and json.loads(man.read_text()).get("inputs_fp") == fp:
        log(f"[replace] already done for these inputs ({fp}) -> {final}")
        return final
    if final.exists():
        log(f"  composite.mp4 was built from other inputs: rebuilding it ({fp})")
        final.unlink()
    finish_mux(src, a.template, final, tpl)
    write_json(man, {"inputs_fp": fp, "source": str(src), "pierre_image": str(a.pierre_image),
                     "basil_image": str(a.basil_image), "strategy": a.strategy,
                     "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return final


def _final_value(ops, x: float, y: float) -> int:
    v = 0
    for x1, y1, x2, y2, val in ops:
        if x1 <= x < x2 and y1 <= y < y2:
            v = val
    return v


def _visible_fraction(ops, box, k: int = 10) -> float:
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return 0.0
    pts = [(x1 + (x2 - x1) * (i + 0.5) / k, y1 + (y2 - y1) * (j + 0.5) / k) for i in range(k) for j in range(k)]
    return sum(1 for x, y in pts if _final_value(ops, x, y) == 0) / len(pts)


def head_preflight(label: str, n: int, ops_fn, target_head, other_heads, max_other: float) -> list[str]:
    """Free geometric check of an isolation input: the other people's HEADS must be hidden in every frame
    (identity is taken from faces; a visible head is what lets Wan's largest-person rule pick the wrong man)."""
    problems, worst, low_t = [], (0.0, -1, ""), 0
    for i in range(n):
        ops = ops_fn(i)
        for name, hb in other_heads(i):
            if hb is None:
                continue
            fr = _visible_fraction(ops, hb)
            if fr > worst[0]:
                worst = (fr, i, name)
            if fr > max_other:
                problems.append(f"frame {i}: {name}'s head {fr:.0%} visible")
        tb = target_head(i)
        if tb is not None and _visible_fraction(ops, tb) < 0.6:
            low_t += 1
    log(f"  pre-flight {label}: other heads max visible {worst[0]:.0%}"
        + (f" (frame {worst[1]}, {worst[2]})" if worst[1] >= 0 else "")
        + f"; frames where the target head is <60% visible: {low_t}")
    return problems


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def detector_preflight(label: str, video: Path, boxes_at, target: str, cfg: Path, weights: Path,
                       step: int) -> list[str]:
    """Optional: simulate Wan2.2-Animate's person choice (open-source pose2d.py, select_type='max', strict,
    shape ratio 0.4) with a YOLOv4 person detector on CPU (needs opencv-python + numpy and the darknet files).
    That fal runs the same rule  # UNVERIFIED.  boxes_at(i) -> {name: box} in the video's pixels. A frame fails
    when the chosen box matches another person's track box better (IoU) than the target's."""
    try:
        import cv2  # noqa: WPS433
        import numpy as np  # noqa: WPS433
    except ImportError:
        log("  WARNING: --preflight-yolo given but opencv-python/numpy are not installed; detector check skipped")
        return []
    net = cv2.dnn.readNetFromDarknet(str(cfg), str(weights))
    ln = net.getUnconnectedOutLayersNames()
    cap = cv2.VideoCapture(str(video))
    i, bad, none, checked = -1, [], 0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if i % step:
            continue
        H, W = fr.shape[:2]
        net.setInput(cv2.dnn.blobFromImage(fr, 1 / 255.0, (608, 608), swapRB=True, crop=False))
        boxes, scores = [], []
        for o in net.forward(ln):
            for d in o:
                sc = d[5:]
                cid = int(np.argmax(sc))
                conf = float(sc[cid]) * float(d[4])
                if cid != 0 or conf < 0.05:
                    continue
                cx, cy, w, h = d[0] * W, d[1] * H, d[2] * W, d[3] * H
                boxes.append([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])
                scores.append(conf)
        keep = cv2.dnn.NMSBoxes(boxes, scores, 0.05, 0.5) if boxes else []
        dets = [(max(0, boxes[k][0]), max(0, boxes[k][1]), min(W, boxes[k][0] + boxes[k][2]),
                 min(H, boxes[k][1] + boxes[k][3])) for k in np.array(keep).flatten()]
        max_shape = max([max(b[2] - b[0], b[3] - b[1]) for b in dets], default=-1)
        best, best_size = None, -1
        for b in dets:
            size, shp = (b[2] - b[0]) * (b[3] - b[1]), max(b[2] - b[0], b[3] - b[1])
            if size > best_size:
                if shp < max_shape * 0.4:
                    continue
                best_size, max_shape, best = size, shp, b
        checked += 1
        if best is None:
            none += 1
            continue
        scores_iou = {k: _iou(best, b) for k, b in boxes_at(i).items()}
        who = max(scores_iou, key=scores_iou.get)
        if who != target:
            bad.append(f"frame {i} ({i / 24:.2f} s): chosen box {best} matches {who} (IoU {scores_iou[who]:.2f}) "
                       f"better than {target} (IoU {scores_iou[target]:.2f})")
    log(f"  pre-flight {label} (YOLOv4 stand-in, every {step} frame(s)): {checked} frames, "
        f"{len(bad)} wrong-person frames, {none} frames with no person (behaviour then  # UNVERIFIED)")
    return bad


def replace_wan_isolate(a, fal: Fal, out: Path, tpl: dict, final: Path) -> Path | None:
    tr = Tracks(a.tracks)
    tr.check_template(tpl)
    n, fps = tpl["frames"], round(tpl["fps"], 3)
    NW = 1280
    NH = even(tpl["height"] * NW / tpl["width"])
    sx, sy = NW / tpl["width"], NH / tpl["height"]
    sxt, syt = tpl["width"] / tr.W, tpl["height"] / tr.H  # tracks -> template pixels
    local = out / "local"
    local.mkdir(exist_ok=True)
    isolate = a.isolate
    if isolate == "head" and not (tr.has_head("pierre") and tr.has_head("basil")):
        log("  WARNING: tracks file has no 'heads' for pierre/basil: falling back to --isolate box")
        isolate = "box"
    def B(name, i, margin):
        b = tr.box(name, i * tr.n / n if n else i)
        return scaled_box(b, sx * sxt, sy * syt, NW, NH, margin)

    def HD(name, i, margin):
        b = tr.head(name, i * tr.n / n if n else i)
        return None if b is None else scaled_box(b, sx * sxt, sy * syt, NW, NH, margin)

    others = [k for k in tr.all_names() if k not in ("pierre", "basil")]
    mg, hm = a.blank_margin, a.head_margin

    # 2) isolation masks (255 = paint grey), painted in order, last op wins.
    def ops_for(target, other, i):
        if isolate == "box":   # legacy rule: hide the others' boxes, never inside the target's box
            return ([B(other, i, mg) + (255,)] + [B(o, i, mg) + (255,) for o in others]
                    + [B(target, i, 0.02) + (0,)])
        th, oh = HD(target, i, hm), HD(other, i, hm)
        oh0 = HD(other, i, 0)
        if target == "pierre":
            # Pierre is in front: hide Basil's WHOLE box (+ others) and give back only Pierre's head. Pierre's
            # body/arm stays visible where it is outside Basil's box. Basil can never be the largest person.
            ops = [B("basil", i, mg) + (255,)] + [B(o, i, mg) + (255,) for o in others]
            ops += [th + (0,)] if th else []
        else:
            # Basil is behind Pierre: hide Pierre's box (+ others), give back Basil's box, hide Pierre's head
            # even where it overlaps Basil's box, then give back Basil's head.
            # The silent third man is left visible by default: Wan keeps the largest person (Basil here), and
            # greying him out made Wan hallucinate junk in the hole (seen in the 4 s test). --hide-others restores it.
            hide = others if getattr(a, "hide_others", False) else []
            ops = ([B("pierre", i, mg) + (255,)] + [B(o, i, mg) + (255,) for o in hide]
                   + [B("basil", i, 0.02) + (0,)])
            ops += [oh + (255,)] if oh else []
            ops += [th + (0,)] if th else []
        # Heads close together: the loose margin of the target's head must not reveal the other head, so the
        # other head's core is hidden LAST (hiding the wrong identity beats showing a sliver more of the target).
        ops += [oh0 + (255,)] if oh0 else []
        return ops

    ops_p = lambda i: ops_for("pierre", "basil", i)  # noqa: E731
    ops_b = lambda i: ops_for("basil", "pierre", i)  # noqa: E731
    # Every derived local file lives in a folder named after the inputs that produced it.
    ops_digest = fingerprint([ops_p(i) for i in range(n)], [ops_b(i) for i in range(n)])
    iso_fp = fingerprint("iso-v3", a.template, ops_digest, a.basil_crop, NW, NH, n, fps)
    iso = local / f"iso_{iso_fp}"
    iso.mkdir(exist_ok=True)
    log(f"  isolation inputs: {iso}  (isolate={isolate})")

    # 1) normalised, silent template
    norm = iso / "norm.mp4"
    if not exists(norm):
        run(["ffmpeg", "-i", str(a.template), "-an", "-vf", f"scale={NW}:{NH}:flags=lanczos,setsar=1,fps={fps}",
             "-frames:v", str(n), "-c:v", "libx264", "-crf", "10", "-preset", "medium", "-pix_fmt", "yuv420p",
             str(norm)])

    m_pin = render_mask(iso / "blank_for_pierre.mkv", NW, NH, n, fps, ops_p)
    m_bin = render_mask(iso / "blank_for_basil.mkv", NW, NH, n, fps, ops_b)

    # 2b) free pre-flight: the other man's head must be hidden in every frame of each pass.
    problems = []
    if tr.has_head("pierre") and tr.has_head("basil"):
        problems += ["Pierre pass: " + p for p in head_preflight(
            "Pierre pass", n, ops_p, lambda i: HD("pierre", i, 0), lambda i: [("basil", HD("basil", i, 0))],
            a.preflight_max_visible)]
        problems += ["Basil pass: " + p for p in head_preflight(
            "Basil pass", n, ops_b, lambda i: HD("basil", i, 0), lambda i: [("pierre", HD("pierre", i, 0))],
            a.preflight_max_visible)]
    else:
        log("  WARNING: no head tracks: head pre-flight skipped")

    def blank(src, mask, dst):
        if exists(dst):
            return dst
        fc = (f"[0:v]format=gbrp[a];color=c=0x7f7f7f:s={NW}x{NH}:r={fps},format=gbrp[b];"
              f"[1:v]format=gray,gblur=sigma=6,format=gbrp[m];[a][b][m]maskedmerge,format=yuv420p")
        run(["ffmpeg", "-i", str(src), "-i", str(mask), "-filter_complex", fc, "-frames:v", str(n), "-r", f"{fps}",
             "-c:v", "libx264", "-crf", "12", "-preset", "medium", str(dst)])
        return dst

    pierre_in = blank(norm, m_pin, iso / "input_pierre.mp4")
    basil_in_full = blank(norm, m_bin, iso / "input_basil_full.mp4")

    # 3) Basil pass gets a static 16:9 zoom window around the union of his boxes (more face pixels).
    crop = None
    if a.basil_crop == "auto":
        xs1, ys1, xs2, ys2 = NW, NH, 0, 0
        for i in range(0, n, 3):
            x1, y1, x2, y2 = B("basil", i, 0.06)
            xs1, ys1, xs2, ys2 = min(xs1, x1), min(ys1, y1), max(xs2, x2), max(ys2, y2)
        cw, ch = xs2 - xs1, ys2 - ys1
        if cw / ch < NW / NH:
            cw = ch * NW / NH
        else:
            ch = cw * NH / NW
        cw, ch = min(NW, even(cw)), min(NH, even(ch))
        cx = min(max(0, even((xs1 + xs2) / 2 - cw / 2)), NW - cw)
        cy = min(max(0, even((ys1 + ys2) / 2 - ch / 2)), NH - ch)
        if cw < NW * 0.92:
            crop = {"x": cx, "y": cy, "w": cw, "h": ch, "zoom": round(NW / cw, 3)}
    write_json(iso / "basil_crop.json", crop or {"zoom": 1.0, "note": "no crop (--basil-crop off or full span)"})
    basil_in = basil_in_full
    if crop:
        basil_in = iso / "input_basil.mp4"
        if not exists(basil_in):
            run(["ffmpeg", "-i", str(basil_in_full), "-vf",
                 f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']},scale={NW}:{NH}:flags=lanczos,setsar=1",
                 "-frames:v", str(n), "-c:v", "libx264", "-crf", "12", "-preset", "medium", "-pix_fmt", "yuv420p",
                 str(basil_in)])
        log(f"  Basil zoom window {crop}")

    # 2c) optional detector pre-flight on the exact videos that would be uploaded.
    if a.preflight_yolo:
        cfg, weights = (Path(p) for p in a.preflight_yolo)

        def all_boxes(i, cropped):
            res = {k: B(k, i, 0.0) for k in tr.all_names()}
            if cropped and crop:
                z = NW / crop["w"]
                res = {k: ((b[0] - crop["x"]) * z, (b[1] - crop["y"]) * z, (b[2] - crop["x"]) * z,
                           (b[3] - crop["y"]) * z) for k, b in res.items()}
            return res
        problems += ["Pierre pass: " + p for p in detector_preflight(
            "Pierre pass", pierre_in, lambda i: all_boxes(i, False), "pierre", cfg, weights, a.preflight_step)]
        problems += ["Basil pass: " + p for p in detector_preflight(
            "Basil pass", basil_in, lambda i: all_boxes(i, True), "basil", cfg, weights, a.preflight_step)]
    if problems:
        for p in problems[:25]:
            log(f"    PRE-FLIGHT FAIL {p}")
        if len(problems) > 25:
            log(f"    ... {len(problems) - 25} more")
        if not a.skip_preflight:
            die(f"pre-flight failed ({len(problems)} problem frames): the wrong man could be replaced. Fix the "
                f"tracks/heads (tracks-preview), raise --blank-margin/--head-margin, or pass --skip-preflight to pay "
                f"anyway. Inputs to inspect: {iso}")
        log("  --skip-preflight given: continuing despite the pre-flight failures")

    # 4) Wan 2.2 Animate replace, once per person. Call names hash the exact uploaded files + parameters.
    chars = {}
    for who, src in (("pierre", Path(a.pierre_image)), ("basil", Path(a.basil_image))):
        chars[who] = fit_16x9_image(src, local / f"char_{who}_{fingerprint(src)}.png")
    billed_s = n / 16.0
    usd = billed_s * WAN_USD_PER_BILLED_SECOND[a.resolution]
    results, call_fp = {}, {}
    for who, vid in (("pierre", pierre_in), ("basil", basil_in)):
        img = chars[who]
        params = {"resolution": a.resolution, "num_inference_steps": a.steps, "guidance_scale": 1, "shift": 5,
                  "seed": a.seed_wan, "video_quality": "maximum", "video_write_mode": "balanced",
                  "return_frames_zip": False}
        call_fp[who] = fingerprint(WAN_REPLACE, vid, img, params)
        args = {"video_url": fal.upload(vid), "image_url": fal.upload(img), **params}
        try:
            r = fal.call(out / "calls", f"wan_replace_{who}_{call_fp[who]}", WAN_REPLACE, args, usd)
        except Refusal as e:
            die(f"Wan replace refused for {who}: {e}. Try --strategy recast, or another reference image.")
        if r is None:
            continue
        results[who] = fal.download(r["video"]["url"], out / f"wan_{who}_{call_fp[who]}_raw.mp4")

    if fal.dry:
        log(f"  [dry-run] local isolation inputs written to {iso} (check input_pierre.mp4 / input_basil.mp4 "
            f"before paying)")
        return None

    # 5) back to the normalised frame grid
    comp_fp = fingerprint("comp-v2", iso_fp, call_fp["pierre"], call_fp["basil"], a.matte, a.matte_margin,
                          a.diff_threshold)
    cdir = local / f"comp_{comp_fp}"
    cdir.mkdir(exist_ok=True)
    ref_dur = n / fps
    p_full = conform(results["pierre"], cdir / "wan_pierre_conf.mp4", n, fps, NW, NH, ref_dur)
    b_conf = conform(results["basil"], cdir / "wan_basil_conf.mp4", n, fps, NW, NH, ref_dur)
    b_full = b_conf
    if crop:
        b_full = cdir / "wan_basil_full.mp4"
        if not exists(b_full):
            fc = (f"[1:v]scale={crop['w']}:{crop['h']}:flags=lanczos[s];[0:v][s]overlay={crop['x']}:{crop['y']}"
                  f":eof_action=pass,format=yuv420p")
            run(["ffmpeg", "-i", str(basil_in_full), "-i", str(b_conf), "-filter_complex", fc, "-frames:v", str(n),
                 "-c:v", "libx264", "-crf", "12", "-preset", "medium", str(b_full)])

    # 6) composite: original -> Basil layer -> Pierre layer (Pierre is in front of Basil).
    #    Matte = (|pass output - pass input| > threshold, dilated) INSIDE the person's box, feathered, and
    #    never where that pass's input was greyed out (the model saw nothing real there).
    #    Areas the model left unchanged (incl. the other man inside the box) fall through to the layer below.
    box_p = render_mask(cdir / "box_pierre.mkv", NW, NH, n, fps, lambda i: [B("pierre", i, a.matte_margin) + (255,)])
    box_b = render_mask(cdir / "box_basil.mkv", NW, NH, n, fps, lambda i: [B("basil", i, a.matte_margin) + (255,)])

    def matte(gen, inp, box, blankmask, dst):
        if exists(dst):
            return dst
        keep = "[3:v]format=gray,gblur=sigma=6,negate[k]"
        if a.matte == "box":
            fc = f"{keep};[2:v]format=gray,gblur=sigma=10[bx];[bx][k]blend=all_mode=multiply"
        else:
            t = int(a.diff_threshold)
            fc = (f"{keep};[0:v]format=gbrp[g];[1:v]format=gbrp[i];[g][i]blend=all_mode=difference,"
                  f"extractplanes=r+g+b[r][gg][b];"
                  f"[r][gg]blend=all_mode=lighten[rg];[rg][b]blend=all_mode=lighten,gblur=sigma=2,"
                  f"lut=y='if(gt(val,{t}),255,0)',gblur=sigma=10,lut=y='if(gt(val,40),255,0)'[d];"
                  f"[2:v]format=gray[bx];[d][bx]blend=all_mode=multiply,gblur=sigma=5[m0];"
                  f"[m0][k]blend=all_mode=multiply")
        run(["ffmpeg", "-i", str(gen), "-i", str(inp), "-i", str(box), "-i", str(blankmask), "-filter_complex", fc,
             "-frames:v", str(n), "-c:v", "ffv1", "-pix_fmt", "gray", str(dst)])
        return dst

    mp = matte(p_full, pierre_in, box_p, m_pin, cdir / "matte_pierre.mkv")
    mb = matte(b_full, basil_in_full, box_b, m_bin, cdir / "matte_basil.mkv")
    comp_norm = cdir / "composite_norm.mp4"
    if not exists(comp_norm):
        fc = ("[0:v]format=gbrp[o];[1:v]format=gbrp[b];[2:v]format=gbrp[mb];[3:v]format=gbrp[p];[4:v]format=gbrp[mp];"
              "[o][b][mb]maskedmerge[c1];[c1][p][mp]maskedmerge,format=yuv420p")
        run(["ffmpeg", "-i", str(norm), "-i", str(b_full), "-i", str(mb), "-i", str(p_full), "-i", str(mp),
             "-filter_complex", fc, "-frames:v", str(n), "-c:v", "libx264", "-crf", "12", "-preset", "medium",
             str(comp_norm)])
    # debug view: composite | mattes
    dbg = cdir / "debug_mattes.mp4"
    if not exists(dbg):
        fc = ("[1:v]format=gray[a];[2:v]format=gray[b];[a][b]hstack,scale=1280:-2,format=yuv420p[m];"
              "[0:v]scale=1280:-2[c];[c][m]vstack,scale=960:-2")
        run(["ffmpeg", "-i", str(comp_norm), "-i", str(mp), "-i", str(mb), "-filter_complex", fc,
             "-frames:v", str(n), "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", str(dbg)])
    shutil.copyfile(dbg, out / "debug_mattes.mp4")
    publish_composite(comp_norm, a, final, tpl, comp_fp)
    log(f"[replace] composite -> {final}   (inspect {out / 'debug_mattes.mp4'} around 8-11 s and 14-17 s)")
    return final


def replace_recast(a, fal: Fal, out: Path, tpl: dict, final: Path) -> Path | None:
    n, fps, dur = tpl["frames"], round(tpl["fps"], 3), tpl["frames"] / tpl["fps"]
    cut = a.recast_split
    if cut < 5 or dur - cut < 5:
        die(f"--recast-split {cut} must leave >=5 s on both sides (clip {dur:.2f} s); recast needs 5-30 s per call")
    f_cut = int(round(cut * fps))
    local = out / "local_recast"
    local.mkdir(exist_ok=True)
    segs = [(0, f_cut), (f_cut, n)]
    pimg = fit_16x9_image(Path(a.pierre_image), local / f"char_pierre_{fingerprint(Path(a.pierre_image))}.png")
    bimg = fit_16x9_image(Path(a.basil_image), local / f"char_basil_{fingerprint(Path(a.basil_image))}.png")
    # Order matters: recast maps photos to the main people LEFT to RIGHT by default -> [Basil, Pierre].
    ref_files = [bimg, pimg]
    refs = [fal.upload(p) for p in ref_files]
    confs, fps_calls = [], []
    for k, (f0, f1) in enumerate(segs):
        seg = local / f"seg{k}_{fingerprint(a.template, f0, f1)}.mp4"
        if not exists(seg):
            # 1280x720 upload, video re-encoded frame-accurately, original audio kept for the model
            run(["ffmpeg", "-i", str(a.template), "-vf",
                 f"trim=start_frame={f0}:end_frame={f1},setpts=PTS-STARTPTS,scale=1280:-2:flags=lanczos",
                 "-af", f"atrim=start={f0/fps}:end={f1/fps},asetpts=PTS-STARTPTS",
                 "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(seg)])
        secs = (f1 - f0) / fps
        params = {"resolution": a.recast_resolution, "prompt": RECAST_PROMPT, "seed": a.seed_wan}
        cfp = fingerprint(RECAST, seg, ref_files, params)
        fps_calls.append(cfp)
        args = {"video_url": fal.upload(seg), "reference_image_urls": refs, **params}
        try:
            r = fal.call(out / "calls", f"recast_seg{k}_{cfp}", RECAST, args,
                         secs * RECAST_USD_PER_SECOND[a.recast_resolution])
        except Refusal as e:
            die(f"recast refused: {e}")
        if r is None:
            continue
        raw = fal.download(r["video"]["url"], out / f"recast_seg{k}_{cfp}_raw.mp4")
        confs.append(conform(raw, local / f"seg{k}_{cfp}_conf.mp4", f1 - f0, fps, tpl["width"], tpl["height"], secs))
    if fal.dry:
        return None
    comp_fp = fingerprint("recast-v2", fps_calls)
    lst = local / f"concat_{comp_fp}.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in confs))
    joined = local / f"joined_{comp_fp}.mp4"
    if not exists(joined):
        run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(joined)])
    return publish_composite(joined, a, final, tpl, comp_fp)


# ----------------------------------------------------------------------------------------------
# Step 3: lipsync (optional)
# ----------------------------------------------------------------------------------------------


def face_point(tr: Tracks, who: str, override: str | None, tpl: dict, default_t: float):
    """-> (frame_number, x, y) in the pixels of the video sent to the model (template size)."""
    if override:
        try:
            xy, _, f = override.partition("@")
            x, y = (int(v) for v in xy.split(","))
            return (int(f) if f else int(default_t * tpl["fps"])), x, y
        except ValueError:
            die(f"bad face spec '{override}', expected X,Y@FRAME")
    sx, sy = tpl["width"] / tr.W, tpl["height"] / tr.H
    if who in tr.faces:
        f, x, y = tr.faces[who]
        return int(f * tpl["frames"] / tr.n), int(x * sx), int(y * sy)
    f = int(default_t * tpl["fps"])
    x1, y1, x2, y2 = tr.box(who, f * tr.n / tpl["frames"])
    return f, int((x1 + x2) / 2 * sx), int((y1 + 0.2 * (y2 - y1)) * sy)


def step_lipsync(a, fal: Fal) -> Path | None:
    out = a.out / "3_lipsync"
    out.mkdir(parents=True, exist_ok=True)
    final = out / "lipsync.mp4"
    video = Path(a.video) if a.video else a.out / "2_replace" / "composite.mp4"
    if not video.is_file():
        if fal.dry:
            log(f"  [dry-run] {video} does not exist yet; planning with the template as stand-in")
            video = a.template
        else:
            die(f"input video not found: {video} (run 'replace' first or pass --video)")
    tpl = ffprobe(a.template)
    n, fps, W, H = tpl["frames"], round(tpl["fps"], 3), tpl["width"], tpl["height"]
    tr = Tracks(a.tracks)
    split, dur = a.split, tpl["duration"]
    fp = face_point(tr, "pierre", a.pierre_face, tpl, 2.0)
    fb = face_point(tr, "basil", a.basil_face, tpl, min(split + 1.0, dur - 0.5))
    run_fp = fingerprint("lip-v2", video, a.template, split, fp, fb, a.sync_auto_detect)
    man = out / "lipsync.json"
    if exists(final) and man.exists() and json.loads(man.read_text()).get("inputs_fp") == run_fp:
        log(f"[lipsync] already done for these inputs ({run_fp}) -> {final}")
        return final
    # Two passes over the FULL clip so durations match (sync_mode cut_off is then harmless):
    # A = audio with Basil's part silenced, target Pierre; B = audio with Pierre's part silenced, target Basil.
    afp = fingerprint(a.template, split)
    aud_p, aud_b = out / f"audio_pierre_only_{afp}.wav", out / f"audio_basil_only_{afp}.wav"
    if not exists(aud_p):
        run(["ffmpeg", "-i", str(a.template), "-vn", "-af", f"volume=enable='gte(t,{split})':volume=0",
             "-c:a", "pcm_s16le", str(aud_p)])
    if not exists(aud_b):
        run(["ffmpeg", "-i", str(a.template), "-vn", "-af", f"volume=enable='lt(t,{split})':volume=0",
             "-c:a", "pcm_s16le", str(aud_b)])
    # The face points are in template pixels/frames: every video sent is first put on the template grid.
    cur = conform(video, out / f"in_{fingerprint(video, n, fps, W, H)}.mp4", n, fps, W, H, n / fps)
    usd = dur / 60 * SYNC_V3_USD_PER_MIN
    for who, aud, (f, x, y) in (("pierre", aud_p, fp), ("basil", aud_b, fb)):
        asd = {"coordinates": [x, y], "frame_number": f}
        if a.sync_auto_detect == "false":
            asd = {"auto_detect": False, **asd}   # UNVERIFIED that this is the switch (see SYNC_V3 notes)
        params = {"sync_mode": "cut_off", "options": {"active_speaker_detection": asd}}
        cfp = fingerprint(SYNC_V3, cur, aud, params)
        args = {"video_url": fal.upload(cur), "audio_url": fal.upload(aud), **params}
        try:
            r = fal.call(out / "calls", f"sync_v3_{who}_{cfp}", SYNC_V3, args, usd)
        except Refusal as e:
            die(f"lipsync refused ({who}): {e}")
        if r is None:
            if who == "pierre":
                log(f"  [dry-run] pass A output would be conformed to {n} f @ {fps} fps {W}x{H} before pass B, so "
                    f"Basil's point {fb[1]},{fb[2]}@{fb[0]} stays valid")
            cur = Path(f"output_of_sync_v3_{who}.mp4")
            continue
        raw = fal.download(r["video"]["url"], out / f"sync_{who}_{cfp}.mp4")
        pi = ffprobe(raw)
        log(f"  sync {who} output: {pi.get('width')}x{pi.get('height')} {pi.get('fps', 0):.3f} fps "
            f"{pi.get('frames')} frames {pi['duration']:.3f} s")
        cur = conform(raw, out / f"sync_{who}_{cfp}_conf.mp4", n, fps, W, H, n / fps)
    if fal.dry:
        return None
    if final.exists():
        final.unlink()
    shutil.copyfile(cur, final)
    write_json(man, {"inputs_fp": run_fp, "source_video": str(video), "source_sha256": file_sha(video),
                     "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    log(f"[lipsync] -> {final}")
    return final


# ----------------------------------------------------------------------------------------------
# Step 4: finish (local only)
# ----------------------------------------------------------------------------------------------


def finish_mux(video: Path, template: Path, dst: Path, tpl: dict | None = None) -> Path:
    """Exact template frame count / fps / size + ORIGINAL template audio (stream copy)."""
    tpl = tpl or ffprobe(template)
    n, fps, W, H = tpl["frames"], round(tpl["fps"], 3), tpl["width"], tpl["height"]
    tmpv = dst.with_name(dst.stem + "_video_only.mp4")
    if tmpv.exists():
        tmpv.unlink()
    conform(video, tmpv, n, fps, W, H, n / fps)
    cmd = ["ffmpeg", "-i", str(tmpv), "-i", str(template), "-map", "0:v:0"]
    if tpl.get("has_audio"):
        cmd += ["-map", "1:a:0", "-c:a", "copy"]
    cmd += ["-c:v", "copy", "-movflags", "+faststart", str(dst)]
    run(cmd)
    tmpv.unlink(missing_ok=True)
    return dst


def pick_finish_input(a, fal: Fal) -> Path | None:
    """--video, else the lipsync output if it was made from the CURRENT composite, else the composite."""
    if a.video:
        return Path(a.video)
    lip, comp = a.out / "3_lipsync" / "lipsync.mp4", a.out / "2_replace" / "composite.mp4"
    if lip.is_file():
        man = lip.with_suffix(".json")
        src_sha = json.loads(man.read_text()).get("source_sha256") if man.exists() else None
        if not comp.is_file() or src_sha == file_sha(comp):
            return lip
        log(f"  WARNING: {lip} was made from an older composite; using the newer {comp} instead "
            f"(rerun 'lipsync' to sync the new one)")
    if comp.is_file():
        return comp
    return None


def step_finish(a, fal: Fal) -> Path:
    out = a.out / "4_final"
    out.mkdir(parents=True, exist_ok=True)
    video = pick_finish_input(a, fal)
    if video is None:
        if not fal.dry:
            die("no input video: pass --video or run replace/lipsync first")
        log(f"  [dry-run] no replace/lipsync output yet; planning with the template as stand-in input")
        video = a.template
    if not video.is_file():
        die(f"input video not found: {video}")
    tpl = ffprobe(a.template)
    dst = out / "final_16x9.mp4"
    log(f"[finish] {video} -> {dst}")
    if fal.dry:
        vi = ffprobe(video)
        log(f"  [dry-run] input {vi.get('width')}x{vi.get('height')} {vi.get('fps', 0):.3f} fps {vi.get('frames')} f; "
            f"would conform to {tpl['frames']} frames @ {tpl['fps']:.3f} fps {tpl['width']}x{tpl['height']} "
            f"(libx264) and mux the template audio ({tpl.get('audio_codec')}, {tpl.get('sample_rate')} Hz, "
            f"{tpl.get('channels')} ch, stream copy) -> {dst}")
        if a.vertical:
            log(f"  [dry-run] would also export {out / 'final_9x16.mp4'} ({a.vertical_size}, "
                f"focus={a.vertical_focus}, centred on the head tracks)")
        return dst
    if dst.exists():
        dst.unlink()  # finish is cheap and local: always rebuild
    finish_mux(video, a.template, dst, tpl)
    verify_output(dst, tpl)
    if a.vertical:
        v = out / "final_9x16.mp4"
        if v.exists():
            v.unlink()
        make_vertical(dst, v, a, tpl)
        verify_output(v, tpl, check_size=False)
    return dst


def make_vertical(src: Path, dst: Path, a, tpl: dict) -> None:
    W, H, fps = tpl["width"], tpl["height"], tpl["fps"]
    cw = min(W, even(H * 9 / 16))
    vw, vh = (int(v) for v in a.vertical_size.split("x"))
    if a.vertical_focus == "center":
        xexpr = f"{(W - cw) // 2}"
    else:
        tr = Tracks(a.tracks)
        sx = W / tr.W

        def cx_of(who, fr):
            """Centre on the HEAD (the body box of the selfie holder widens when his arm reaches out);
            falls back to the body box when the head is out of frame or not tracked."""
            hb = tr.head(who, fr)
            if hb is not None:
                return (hb[0] + hb[2]) / 2 * sx
            if who == "pierre":   # the camera holder's head is always at the right edge
                b = tr.box(who, fr)
                return (max(b[0], b[2] - cw / sx) + b[2]) / 2 * sx
            b = tr.box(who, fr)
            return (b[0] + b[2]) / 2 * sx

        pts = []
        for f in range(0, tpl["frames"], 3):
            t = f / fps
            fr = f * tr.n / tpl["frames"]
            who = "pierre"
            if a.vertical_focus == "speaker" and t >= a.split:
                who = "basil"
            cxp = cx_of(who, fr)
            if a.vertical_focus == "speaker" and abs(t - a.split) < 0.4:  # ease between speakers
                u = (t - (a.split - 0.4)) / 0.8
                cxp = cx_of("pierre", fr) * (1 - u) + cx_of("basil", fr) * u
            pts.append([t, cxp])
        sm = []  # moving average to avoid jitter
        for i in range(len(pts)):
            win = [p[1] for p in pts[max(0, i - 2): i + 3]]
            sm.append((pts[i][0], min(max(0, sum(win) / len(win) - cw / 2), W - cw)))
        xexpr = piecewise_expr(sm)
    if vw > cw * 1.5:
        log(f"  note: 9:16 crop is {cw}x{H} upscaled to {vw}x{vh} (x{vw / cw:.2f}); use --vertical-size "
            f"{even(cw)}x{even(H)} to avoid upscaling")
    vf = f"crop={cw}:{H}:'{xexpr}':0,scale={vw}:{vh}:flags=lanczos,setsar=1"
    run(["ffmpeg", "-i", str(src), "-filter_complex", f"[0:v]{vf}[v]", "-map", "[v]", "-map", "0:a?",
         "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "copy",
         "-movflags", "+faststart", str(dst)])
    log(f"  9:16 -> {dst} (focus={a.vertical_focus})")


def verify_output(path: Path, tpl: dict, check_size: bool = True) -> None:
    o = ffprobe(path)
    problems = []
    if o.get("frames") != tpl["frames"]:
        problems.append(f"frames {o.get('frames')} != {tpl['frames']}")
    if abs(o.get("fps", 0) - tpl["fps"]) > 0.01:
        problems.append(f"fps {o.get('fps')} != {tpl['fps']}")
    if check_size and (o.get("width"), o.get("height")) != (tpl["width"], tpl["height"]):
        problems.append(f"size {o.get('width')}x{o.get('height')}")
    if tpl.get("has_audio") and not o.get("has_audio"):
        problems.append("no audio")
    if abs(o["duration"] - tpl["duration"]) > 0.05:
        problems.append(f"duration {o['duration']:.3f} != {tpl['duration']:.3f}")
    status = "OK" if not problems else "MISMATCH: " + "; ".join(problems)
    log(f"  verify {path.name}: {o.get('width')}x{o.get('height')} {o.get('fps', 0):.3f} fps {o.get('frames')} frames "
        f"{o['duration']:.3f} s audio={o.get('audio_codec')} -> {status}")
    if problems:
        die(f"output check failed for {path}")


# ----------------------------------------------------------------------------------------------
# tracks-preview (local)
# ----------------------------------------------------------------------------------------------


def step_tracks_preview(a, fal: Fal) -> Path:
    out = a.out / "0_tracks_preview"
    out.mkdir(parents=True, exist_ok=True)
    tpl = ffprobe(a.template)
    tr = Tracks(a.tracks)
    tr.check_template(tpl)
    n, fps, W, H = tpl["frames"], round(tpl["fps"], 3), tpl["width"], tpl["height"]
    sx, sy = W / tr.W, H / tr.H
    colors = {"pierre": "0xff3030", "basil": "0x30ff30"}
    masks = []
    for name in tr.all_names():
        def ops(i, name=name):
            b = scaled_box(tr.box(name, i * tr.n / n), sx, sy, W, H)
            res = [b + (110,)]
            hb = tr.head(name, i * tr.n / n) if name in tr.heads else None
            if hb is not None:   # head box outline (used by isolation and the 9:16 crop)
                x1, y1, x2, y2 = scaled_box(hb, sx, sy, W, H)
                res += [(x1, y1, x2, min(y2, y1 + 4), 255), (x1, max(y1, y2 - 4), x2, y2, 255),
                        (x1, y1, min(x2, x1 + 4), y2, 255), (max(x1, x2 - 4), y1, x2, y2, 255)]
            if name in tr.faces:
                f, x, y = tr.faces[name]
                if abs(i - f * n / tr.n) < 12:
                    x, y = int(x * sx), int(y * sy)
                    res.append((x - 8, y - 8, x + 8, y + 8, 255))
            return res
        mp = out / f"mask_{name}.mkv"
        if mp.exists():
            mp.unlink()
        masks.append((name, render_mask(mp, W, H, n, fps, ops)))
    ins, fc, last = ["-i", str(a.template)], ["[0:v]format=gbrp[c0]"], "c0"
    for k, (name, mp) in enumerate(masks, 1):
        ins += ["-i", str(mp)]
        col = colors.get(name, "0x3080ff")
        fc.append(f"color=c={col}:s={W}x{H}:r={fps},format=gbrp[k{k}];[{k}:v]format=gbrp[m{k}];"
                  f"[{last}][k{k}][m{k}]maskedmerge[c{k}]")
        last = f"c{k}"
    dst = out / "tracks_preview.mp4"
    run(["ffmpeg"] + ins + ["-filter_complex", ";".join(fc) + f";[{last}]format=yuv420p[v]", "-map", "[v]",
                            "-frames:v", str(n), "-c:v", "libx264", "-crf", "23", str(dst)])
    sheet = out / "tracks_preview_sheet.jpg"
    sel = "+".join(f"eq(n\\,{int(n*k/6)})" for k in range(6))
    run(["ffmpeg", "-i", str(dst), "-vf", f"select='{sel}',scale=640:-2,tile=2x3", "-frames:v", "1", str(sheet)])
    log(f"[tracks-preview] red=Pierre slot, green=Basil slot, blue=others, white outlines=head boxes, "
        f"white squares=lip-sync face points\n"
        f"  {dst}\n  {sheet}")
    return dst


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    g = common.add_argument_group("global")
    g.add_argument("--out", type=Path, default=Path("pipeline_out"), help="output/work directory (default ./pipeline_out)")
    g.add_argument("--dry-run", action="store_true", help="no network, no key: validate inputs, print planned API calls")
    g.add_argument("--yes", action="store_true", help="do not ask before paid calls")
    g.add_argument("--max-wait", type=float, default=DEFAULT_MAX_WAIT_S,
                   help="max seconds to wait for one queued fal call; the request_id is kept so a rerun resumes it")

    def add_template(p, required=True):
        p.add_argument("--template", type=Path, required=required, help="template reel mp4 (reel 1)")
        p.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS,
                       help=f"person keyframe boxes JSON (default: {DEFAULT_TRACKS.name} next to this script)")

    def add_refs(p):
        p.add_argument("--pierre-refs", nargs="+", default=[], metavar="IMG[@x,y,w,h]",
                       help="Pierre photos, identity anchor FIRST (frontal), then 3/4 smile, then daylight")
        p.add_argument("--basil-refs", nargs="+", default=[], metavar="IMG[@x,y,w,h]",
                       help="Basil photos; crop group photos with @x,y,w,h so only Basil remains")
        p.add_argument("--who", default="pierre,basil")
        p.add_argument("--variants", type=int, default=3, help="variants per person (2-4)")
        p.add_argument("--ref-models", default=DEFAULT_REF_CHAIN, help=f"fallback chain, known: {','.join(REF_MODELS)}")
        p.add_argument("--seed", type=int, default=1234)
        p.add_argument("--pierre-desc", default=PEOPLE_DEFAULT_DESC["pierre"])
        p.add_argument("--basil-desc", default=PEOPLE_DEFAULT_DESC["basil"])

    def add_replace(p):
        p.add_argument("--pierre-image", help="chosen step-1 image of Pierre in the suit")
        p.add_argument("--basil-image", help="chosen step-1 image of Basil in the suit")
        p.add_argument("--strategy", choices=["wan-isolate", "recast"], default="wan-isolate")
        p.add_argument("--resolution", choices=["480p", "580p", "720p"], default="720p")
        p.add_argument("--steps", type=int, default=20, help="Wan num_inference_steps (try 30 if faces are soft)")
        p.add_argument("--wan-seed", dest="seed_wan", type=int, default=42)
        p.add_argument("--basil-crop", choices=["auto", "off"], default="auto")
        p.add_argument("--hide-others", action="store_true", help="also grey out the third man in Basil's pass (old behaviour)")
        p.add_argument("--blank-margin", type=float, default=0.06, help="box growth when hiding people")
        p.add_argument("--isolate", choices=["head", "box"], default="head",
                       help="head (default): hide the other man's whole box, give back only the target's head "
                            "(Pierre pass) / also hide the other man's head (Basil pass). box: legacy rule")
        p.add_argument("--head-margin", type=float, default=0.15, help="head box growth for isolation")
        p.add_argument("--preflight-max-visible", type=float, default=0.10,
                       help="max visible fraction of the other man's head in any frame before refusing to pay")
        p.add_argument("--preflight-yolo", nargs=2, metavar=("CFG", "WEIGHTS"),
                       help="optional: YOLOv4 darknet cfg+weights to simulate Wan's largest-person choice on the "
                            "isolation inputs (needs opencv-python + numpy); wrong-person frames block the run")
        p.add_argument("--preflight-step", type=int, default=3, help="analyse every Nth frame with --preflight-yolo")
        p.add_argument("--skip-preflight", action="store_true", help="pay even if a pre-flight check fails")
        p.add_argument("--matte", choices=["diff", "box"], default="diff")
        p.add_argument("--matte-margin", type=float, default=0.10, help="box growth for compositing")
        p.add_argument("--diff-threshold", type=int, default=22, help="0-255, for --matte diff")
        p.add_argument("--recast-split", type=float, default=8.6, help="cut point (s) for --strategy recast")
        p.add_argument("--recast-resolution", choices=["768P", "1080P"], default="768P")

    def add_lipsync(p, with_video=True):
        if with_video:
            p.add_argument("--video", help="input video (default: out/2_replace/composite.mp4)")
        p.add_argument("--split", type=float, default=14.2, help="time (s) where Basil starts speaking")
        p.add_argument("--pierre-face", help="X,Y@FRAME on Pierre's face (template pixels); default from tracks")
        p.add_argument("--basil-face", help="X,Y@FRAME on Basil's face (template pixels); default from tracks")
        p.add_argument("--sync-auto-detect", choices=["false", "omit"], default="false",
                       help="send active_speaker_detection.auto_detect=false with the face point (default), or omit "
                            "the key (UNVERIFIED which one makes sync v3 use the point)")

    def add_finish(p, with_video=True):
        if with_video:
            p.add_argument("--video", help="generated video (default: lipsync output, else replace output)")
        p.add_argument("--vertical", action="store_true", help="also export a 9:16 crop")
        p.add_argument("--vertical-focus", choices=["center", "pierre", "speaker"], default="speaker")
        p.add_argument("--vertical-size", default="720x1280")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("refs", parents=[common], help="step 1: suit reference portraits (then stop)")
    add_refs(p)
    p = sub.add_parser("replace", parents=[common], help="step 2: character replacement")
    add_template(p)
    add_replace(p)
    p = sub.add_parser("lipsync", parents=[common], help="step 3 (optional): lip-sync repair")
    add_template(p)
    add_lipsync(p)
    p = sub.add_parser("finish", parents=[common], help="step 4: original audio + exports (local)")
    add_template(p)
    add_finish(p)
    p.add_argument("--split", type=float, default=14.2, help="speaker change time (s) for --vertical-focus speaker")
    p = sub.add_parser("all", parents=[common], help="all steps (stops after refs if no images chosen)")
    add_template(p)
    add_refs(p)
    add_replace(p)
    add_lipsync(p, with_video=False)
    add_finish(p, with_video=False)
    p.add_argument("--lipsync", action="store_true", help="include the optional lip-sync pass")
    p = sub.add_parser("tracks-preview", parents=[common], help="local: draw the person tracks on the template")
    add_template(p)
    return ap


def validate_common(a) -> None:
    need_tool("ffmpeg")
    need_tool("ffprobe")
    if getattr(a, "template", None) is not None:
        if not a.template.is_file():
            die(f"template not found: {a.template}")
    if getattr(a, "tracks", None) is not None and a.cmd in ("replace", "all", "tracks-preview", "lipsync") \
            and not (a.cmd == "replace" and getattr(a, "strategy", "") == "recast"):
        if not a.tracks.is_file():
            die(f"tracks file not found: {a.tracks}")
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        validate_common(a)
        fal = Fal(a.out, a.dry_run, a.yes, a.max_wait)
        if a.dry_run:
            log("=== DRY RUN: no network calls, FAL_KEY not needed; local preparation still runs ===")
        elif a.cmd in ("refs", "replace", "lipsync", "all") and not os.environ.get("FAL_KEY"):
            fal.client()  # fails early with a clear message, before any local work
        if a.cmd == "refs":
            step_refs(a, fal)
        elif a.cmd == "replace":
            step_replace(a, fal)
        elif a.cmd == "lipsync":
            step_lipsync(a, fal)
        elif a.cmd == "finish":
            step_finish(a, fal)
        elif a.cmd == "tracks-preview":
            step_tracks_preview(a, fal)
        elif a.cmd == "all":
            if a.pierre_refs or a.basil_refs:
                step_refs(a, fal)
            if not (a.pierre_image and a.basil_image):
                log("[all] stopping after refs: pass --pierre-image and --basil-image (your picks) to continue.")
                return 0
            step_replace(a, fal)
            a.video = None
            if a.lipsync:
                step_lipsync(a, fal)
            step_finish(a, fal)
        if fal.planned_usd:
            log(f"\nEstimated fal spend for the calls {'planned' if a.dry_run else 'made'} in this run: "
                f"~${fal.planned_usd:.2f} (estimates from fal pages, re-check)")
        return 0
    except PipelineError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted (rerun the same command to resume)", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())

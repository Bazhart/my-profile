"""OmniHuman v1.5 on fal: image + audio (+ speaker mask) -> talking video."""
import json, sys, pathlib
import fal_client

out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
image, audio, mask, prompt_file = sys.argv[2:6]
resolution = sys.argv[6] if len(sys.argv) > 6 else "720p"
args = {
    "image_url": fal_client.upload_file(image),
    "audio_url": fal_client.upload_file(audio),
    "prompt": pathlib.Path(prompt_file).read_text().strip(),
    "resolution": resolution,
}
if mask != "-":
    args["mask_url"] = fal_client.upload_file(mask)
(out / "request.json").write_text(json.dumps({**args, "_files": [image, audio, mask]}, indent=1))
res = fal_client.subscribe("fal-ai/bytedance/omnihuman/v1.5", arguments=args, with_logs=False)
(out / "response.json").write_text(json.dumps(res, indent=1))

def urls(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "url" and isinstance(v, str):
                yield v
            else:
                yield from urls(v)
    elif isinstance(o, list):
        for v in o:
            yield from urls(v)

for u in urls(res):
    print(u)

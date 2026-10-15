"""Generic fal call: image + audio -> talking video. Usage: avatar.py OUTDIR ENDPOINT IMAGE AUDIO PROMPT_FILE"""
import json, sys, pathlib
import fal_client

out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
endpoint, image, audio, prompt_file = sys.argv[2:6]
args = {
    "image_url": fal_client.upload_file(image),
    "audio_url": fal_client.upload_file(audio),
    "prompt": pathlib.Path(prompt_file).read_text().strip(),
}
(out / "request.json").write_text(json.dumps({"endpoint": endpoint, **args, "_files": [image, audio]}, indent=1))
res = fal_client.subscribe(endpoint, arguments=args, with_logs=False)
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

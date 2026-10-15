import json, sys, pathlib, fal_client
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
video, audio = sys.argv[2:4]
args = {"video_url": fal_client.upload_file(video), "audio_url": fal_client.upload_file(audio)}
res = fal_client.subscribe("fal-ai/latentsync", arguments=args, with_logs=False)
(out / "response.json").write_text(json.dumps(res, indent=1))
print(res["video"]["url"])

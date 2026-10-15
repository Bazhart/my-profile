import json, sys, pathlib, fal_client
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
scene, basil = sys.argv[2:4]
prompt = ("Image 1 is a frame from a handheld selfie video in a modern law office. Create the next shot of the same video: "
  "the phone camera has turned toward the curly-haired man with a moustache in the navy suit seated across the table, "
  "who is now the main subject, centered and closer to the camera, facing the camera, relaxed and confident, about to speak. "
  "His face and identity must exactly match image 2 (wavy light-brown hair with volume on top and short faded sides, thin moustache "
  "and short chin beard, brown eyes), lean fit athletic build, tailored dark navy two-piece Italian suit, crisp white shirt, "
  "top button open, no tie. The bald man in the navy suit and purple tie is still seated beside him, slightly out of focus. "
  "The blond man in the foreground is only partly visible at the right edge of the frame, out of focus. Same office, same "
  "American flag wall, same lighting, color grade and wide-angle phone-camera look as image 1. No phone visible. Photorealistic.")
args = {"prompt": prompt, "image_urls": [fal_client.upload_file(scene), fal_client.upload_file(basil)],
        "aspect_ratio": "16:9", "resolution": "2K", "num_images": 2, "output_format": "png"}
res = fal_client.subscribe("fal-ai/nano-banana-pro/edit", arguments=args, with_logs=False)
(out / "response.json").write_text(json.dumps(res, indent=1))
for im in res.get("images", []): print(im["url"])

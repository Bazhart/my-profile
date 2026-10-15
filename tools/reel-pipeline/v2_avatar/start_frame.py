"""Generate the start frame: reel-1 diner frame with the two friends swapped in (Nano Banana Pro edit)."""
import json, sys, pathlib
import fal_client

out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
frame, pierre, basil = sys.argv[2:5]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 3

SCENES = {
"diner": (
    "Edit image 1, a wide-angle handheld selfie video frame shot inside a retro American diner. "
    "Replace the man in the foreground on the right who is holding the phone (cap, cream polo) with the man from image 2: "
    "exact same face and identity as image 2, blond hair slicked back and tied in a short ponytail, no cap, "
    "lean fit athletic build, warm natural smile, talking to the camera. "
    "Replace the man seated across the counter in the middle (dark tracksuit) with the man from image 3: "
    "exact same face and identity as image 3, wavy light-brown hair with volume on top and short faded sides, "
    "thin moustache and short chin beard, lean fit athletic build, relaxed, looking toward the man in the foreground. "
    "Both men are young, fit and good-looking, not heavy-set. "
    "Both wear tailored dark navy blue two-piece Italian suits, slim fit, with crisp white dress shirts, "
    "top button open, no tie. "
    "Keep everything else from image 1 exactly the same: the diner interior, the counter, the ceiling lights, "
    "the man in the blue suit further back, the camera angle, the wide-angle lens perspective, the framing and the composition. "
    "Photorealistic, natural skin texture, same lighting, color grade and film grain as image 1, "
    "looks like a real frame from the same video."
),
"suits": (
    "Edit image 1, a wide-angle handheld selfie video frame shot at a table in a modern corporate law office. "
    "Replace the man in the foreground on the right who is holding the phone with the man from image 2: "
    "exact same face and identity as image 2, blond hair slicked back and tied in a short ponytail, "
    "lean fit athletic build, warm confident smile, talking to the camera. "
    "Replace the man seated across the table in the middle (grey suit, slicked-back hair) with the man from image 3: "
    "exact same face and identity as image 3, wavy light-brown hair with volume on top and short faded sides, "
    "thin moustache and short chin beard, lean fit athletic build, relaxed and confident, looking toward the camera. "
    "Both men are young, fit and good-looking. "
    "Both wear tailored dark navy blue two-piece Italian suits, slim fit, with crisp white dress shirts, "
    "top button open, no tie. "
    "Keep the man on the far left in the navy suit and purple tie unchanged. "
    "Keep everything else from image 1 exactly the same: the office interior, the table, the lights, the American flag "
    "on the wall, the camera angle, the wide-angle lens perspective, the framing and the composition. "
    "Photorealistic, natural skin texture, same lighting, color grade and grain as image 1, "
    "looks like a real frame from the same video."
),
}
scene = sys.argv[6] if len(sys.argv) > 6 else "diner"
PROMPT = SCENES[scene]

urls = [fal_client.upload_file(p) for p in (frame, pierre, basil)]
args = {"prompt": PROMPT, "image_urls": urls, "aspect_ratio": "16:9", "resolution": "2K",
        "num_images": n, "output_format": "png"}
res = fal_client.subscribe("fal-ai/nano-banana-pro/edit", arguments=args, with_logs=False)
(out / "request.json").write_text(json.dumps({**args, "image_urls": [str(p) for p in (frame, pierre, basil)]}, indent=1))
(out / "response.json").write_text(json.dumps(res, indent=1))
for i, im in enumerate(res.get("images", []), 1):
    print(im["url"])

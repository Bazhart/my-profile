# v2 : image de départ + avatar parlant (la méthode qui a marché)

Le remplacement de personnage de `run_pipeline.py` (Wan 2.2 Animate) donnait des artefacts : trous remplis au hasard, visages en double. Cette version génère la scène au lieu de retoucher le reel.

1. **Image de départ** : `start_frame.py OUT FRAME PIERRE_IMG BASIL_IMG N suits`, avec Nano Banana Pro edit.
   - `FRAME` : une image du reel modèle.
   - Ne jamais demander d'effacer un filigrane : Google refuse la requête.
2. **Plan A** (Pierre parle) : `avatar.py OUT fal-ai/kling-video/ai-avatar/v2/pro START.jpg AUDIO_0-14.2s.wav prompt_A.txt`
3. **Image cadrée sur Basil** : `basil_frame.py OUT START.png BASIL_IMG`, puis plan B avec `avatar.py` et `prompt_B.txt` sur l'audio 14.2-17.1 s.
4. **Montage** (ffmpeg) : les 426 premières images du plan A et les 85 premières du plan B (30 fps), concaténées, puis l'audio d'origine complet du reel.
   - Le décalage audio mesuré entre les sorties Kling et l'audio fourni est de 0,000 s.

`omni.py` (OmniHuman v1.5) renvoyait « downstream_service_unavailable » à chaque essai, même sans masque et avec une petite image.

La clé est lue dans `FAL_KEY`, uniquement par `fal-client`. Aucune photo ni vidéo n'est versionnée.

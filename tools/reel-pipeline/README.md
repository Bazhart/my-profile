# Pipeline reel « diner » : Pierre et Basil en costume (fal.ai)

`run_pipeline.py` reprend le reel 1 comme modèle et remplace les deux personnages par Pierre (homme au premier plan qui filme) et Basil (homme assis au comptoir, en survêtement). Tous les deux portent un costume bleu nuit et une chemise blanche. L'audio d'origine du reel est conservé.

Le dossier ne contient aucune photo ni vidéo personnelle : tous les chemins se passent en arguments. Le `.gitignore` exclut les médias.

## Prérequis

```bash
pip install -r requirements.txt        # fal-client==1.0.3
ffmpeg -version                        # ffmpeg + ffprobe dans le PATH
export FAL_KEY=...                     # la clé fal.ai, uniquement par variable d'environnement
```

Le script ne lit la clé que dans `FAL_KEY`. Il ne l'affiche jamais et ne l'écrit dans aucun fichier.

Hôtes à autoriser dans l'allowlist réseau, par ordre d'importance :

1. `queue.fal.run` : soumission, statut et résultat des appels (fal-client 1.0.3).
2. `rest.fal.ai` : jeton d'upload et upload de secours via GCS.
3. `v3b.fal.media` et `v3.fal.media` : upload des fichiers et téléchargement des résultats. Le plus simple est `*.fal.media`.
4. `queue.falrun.com` : domaine de secours que fal-client essaie si `queue.fal.run` ne répond pas.
5. `fal.run` et `falrun.com` : pas utilisés par le script (il passe par la file d'attente), mais sans risque à ajouter.
6. `storage.googleapis.com` : seulement si fal renvoie l'upload vers GCS (chemin de secours, non vérifié en pratique).
7. `pypi.org` et `files.pythonhosted.org` : pour `pip install`.

Si un hôte est bloqué, le script s'arrête en affichant son nom exact (« cannot reach host '…' »), qu'il s'agisse d'un upload, d'une soumission, d'un statut ou d'un téléchargement. Ajoute-le à l'allowlist et relance la même commande : rien n'est repayé. Le fichier `pipeline_out/hosts_seen.txt` liste en plus les hôtes des URL renvoyées par fal.

## Les options communes

- `--dry-run` : aucun appel réseau, pas besoin de clé. Le script vérifie les fichiers, prépare tout en local, puis affiche chaque appel prévu (endpoint et arguments) avec son coût estimé.
- `--out DIR` : dossier de travail (`./pipeline_out` par défaut). Tous les JSON de requête et de réponse et tous les fichiers téléchargés y sont enregistrés.
- `--yes` : pas de confirmation avant les appels payants. Sans cette option, le script demande `y/N` avant chaque appel.
- `--max-wait SECONDES` : attente maximale d'un appel fal (3600 s par défaut). Une fois ce délai atteint, le script s'arrête mais garde le `request_id`.

Le pipeline peut reprendre là où il s'est arrêté :

- **Une requête déjà soumise** est reprise grâce à son `request_id` (fichier `<appel>.pending.json`), sans être relancée ni repayée. Ce fichier n'est supprimé qu'une fois la requête vraiment terminée, avec un résultat ou un refus définitif. Après un plantage, une coupure réseau, une erreur HTTP passagère pendant l'attente ou un dépassement de `--max-wait`, relance simplement la même commande. Le `request_id` est aussi recopié dans les fichiers `*.error.json` et `*.poll_error.json`.
- **Les fichiers et appels en cache sont nommés d'après leurs entrées** : contenu des photos et paramètres. Relancer avec les mêmes entrées réutilise ce qui est déjà fait. Changer la photo, une marge ou un seuil produit de nouveaux fichiers et de nouveaux appels, payants. Revenir ensuite aux anciennes entrées réutilise les anciens résultats sans repayer.
- **Les fichiers déjà uploadés** sont réutilisés pendant 6 jours, puis uploadés de nouveau, car fal les supprime après 7 jours.

## Étape 0 (gratuite) : vérifier le suivi des personnes

```bash
python3 run_pipeline.py tracks-preview --template reel1_modele_diner.mp4
```

Le fichier `reel1_tracks.json` contient les boîtes de chaque personne image par image. Elles ont été détectées automatiquement sur le reel 1. Cette commande les dessine sur la vidéo pour que tu puisses les contrôler :

- rouge : Pierre
- vert : Basil
- bleu : l'homme en costume bleu

Si tu changes de modèle de vidéo, il faut fournir un autre fichier de boîtes avec `--tracks`.

## Étape 1 : photos de référence en costume

```bash
python3 run_pipeline.py refs --out pipeline_out \
  --pierre-refs pierre_face_snap_FACE_PRINCIPALE.png pierre_face_cuisine_3-4.png pierre_face_lunettes_1.png \
  --basil-refs basil_face_selfie_FACE_PRINCIPALE.jpg "basil_face_chemise_blanche.png@0,480,420,640" basil_face_casquette_bateau.png \
  --variants 3
```

Les deux listes de photos fonctionnent ainsi :

- **Pierre :** mets d'abord la photo de face, même s'il y fait la moue. C'est l'ancre d'identité. Ajoute ensuite la photo 3/4 souriante, puis celle en plein jour. Le prompt demande explicitement de ne pas reprendre la moue, la contre-plongée ni la lumière dure. Il demande aussi une expression neutre, les lèvres fermées et la caméra à hauteur des yeux.
- **Basil :** la mention `@x,y,w,h` recadre une photo de groupe pour ne garder que lui. Le cadrage `0,480,420,640` vaut pour `basil_face_chemise_blanche.png`.

Si tu changes de photo ou de description, le script le détecte et produit de nouvelles images. Les anciennes sont conservées.

Les modèles sont essayés dans cet ordre, chacun ne servant que si le précédent refuse :

1. `fal-ai/nano-banana-pro/edit`
2. `bytedance/seedream/v5/pro/edit`
3. `fal-ai/flux-2-pro/edit`

Le script passe au modèle suivant dans trois cas : une erreur 422, une réponse sans image, ou toute autre erreur HTTP définitive (400, 500…). On ne sait pas sous quelle forme Google refuse via fal. Seules les erreurs 401 et 403 (clé ou compte) arrêtent tout.

Quelques détails sur ce passage d'un modèle à l'autre :

- Les images déjà obtenues sont toujours gardées. Si un modèle en a fourni au moins une, le script ne paie pas le suivant.
- Un modèle qui a refusé est noté dans `refused_<hash>.json` et n'est plus rappelé pour les mêmes entrées. Supprime ce fichier pour le retenter.

**Le script s'arrête ensuite.** Ouvre `pipeline_out/1_refs/<pierre|basil>/contact_sheet.jpg` et choisis la meilleure image : ressemblance d'abord, pas de moue, costume correct. L'étape 2 recopie fidèlement ce visage. Si aucune image ne convient, relance avec un autre `--seed` : de nouvelles images sont produites à côté des anciennes.

## Étape 2 : remplacement des personnages

```bash
python3 run_pipeline.py replace --out pipeline_out --template reel1_modele_diner.mp4 \
  --pierre-image pipeline_out/1_refs/pierre/<choix>.png \
  --basil-image  pipeline_out/1_refs/basil/<choix>.png
```

La stratégie par défaut s'appelle `wan-isolate`. Elle passe par `fal-ai/wan/v2.2-14b/animate/replace`, appelé deux fois.

Cet endpoint ne permet pas de choisir la personne à remplacer. Il prend la plus grande personne de chaque image, et cette personne change vers 14,9-16,4 s, quand Pierre se détourne. Le script contourne cette limite ainsi (`--isolate head`, le mode par défaut) :

1. **Passe Pierre :** toute la boîte de Basil et celle de l'homme en bleu sont recouvertes d'un aplat gris. Seule la tête de Pierre est rendue visible. Basil n'est donc jamais visible, même quand il est dans la boîte de Pierre. En contrepartie, la partie du bras de Pierre qui passe devant Basil est aussi grisée.
2. **Passe Basil :** la boîte de Pierre est masquée, ainsi que sa tête, même là où elle chevauche la boîte de Basil. L'image est recadrée autour de Basil (zoom fixe d'environ ×1,26).
3. **Contrôle avant paiement (gratuit) :** pour chaque image, le script vérifie que la tête de l'autre homme est cachée. Si elle reste visible à plus de 10 % dans une seule image, il refuse de payer. Option `--preflight-yolo CFG WEIGHTS` (nécessite `opencv-python` et `numpy`) : un détecteur YOLOv4 simule en plus le choix de Wan (« plus grande personne ») sur les vidéos exactes qui seraient envoyées. Le script refuse de payer si une seule image désigne le mauvais homme. `--skip-preflight` force le passage.
4. **Composition :** les deux résultats sont replacés sur l'original avec un masque de différence. Ce masque est limité à la boîte de chaque personne et aux zones que la passe a réellement vues (jamais les zones grisées). Pierre est placé au-dessus de Basil.
5. **Calage final :** le résultat est ramené à 409 images à 24 i/s, au format 1276x718, et l'audio d'origine est remis.

Les têtes sont suivies dans `reel1_tracks.json` (clé `heads`, placées à l'œil). L'ancien mode `--isolate box` reste disponible, mais il laissait Basil visible dans la passe de Pierre. Ce défaut est visible entre 15,9 et 16,5 s, et en partie entre 8 et 11 s.

Avant de payer, lance `--dry-run` et regarde les vidéos `2_replace/local/iso_<hash>/input_pierre.mp4` et `input_basil.mp4` (le chemin exact est affiché). Après le rendu, contrôle `2_replace/debug_mattes.mp4` autour de 8-11 s et de 14-17 s.

Il existe une stratégie de secours en un seul passage : `--strategy recast` (`minimax/h3-max/recast`). Elle traite les deux personnes et les costumes en une fois, mais découpe la vidéo en deux plans de 0-8,6 s et 8,6-17,13 s.

Recast attribue par défaut les photos aux personnes de gauche à droite. Le script envoie donc Basil en premier (à gauche, en face) et Pierre en second (à droite, au premier plan), et le prompt dit la même chose. L'attribution ne dépend donc pas d'une consigne du prompt dont l'effet n'est pas vérifié.

## Étape 3 (optionnelle) : corriger le lip-sync

Ne la lance que si les lèvres décrochent.

```bash
python3 run_pipeline.py lipsync --out pipeline_out --template reel1_modele_diner.mp4
```

Elle utilise `fal-ai/sync-lipsync/v3` en deux passes sur toute la durée du clip :

1. **Pierre :** l'audio de Basil (14,2-17,1 s) est coupé et le modèle vise le visage de Pierre.
2. **Basil :** l'audio de Pierre (0-14,2 s) est coupé et le modèle vise le visage de Basil.

Chaque visage est désigné par un point (x, y) sur une image donnée, en pixels du modèle. Les points par défaut sont dans `reel1_tracks.json`. Pour les changer, utilise `--pierre-face X,Y@FRAME` et `--basil-face X,Y@FRAME`.

Avant chaque passe, la vidéo envoyée est recalée sur la grille du modèle (409 images, 24 i/s, 1276x718). Le point de Basil reste donc valable même si la sortie de la passe Pierre change de taille ou de cadence. Le journal affiche la taille et la cadence réelles de chaque sortie.

Le script envoie `auto_detect: false` avec le point. Ce n'est pas vérifié. `--sync-auto-detect omit` envoie le point sans cette clé.

## Étape 4 : finalisation (locale et gratuite)

```bash
python3 run_pipeline.py finish --out pipeline_out --template reel1_modele_diner.mp4 --vertical
```

Cette étape prend la sortie du lip-sync si elle a été faite à partir du composite actuel. Sinon, elle prend le composite de l'étape 2 et prévient. On peut aussi imposer une vidéo avec `--video`. En `--dry-run` sans sortie précédente, le modèle sert de vidéo de test.

Elle produit :

- `4_final/final_16x9.mp4` : H.264, exactement le même nombre d'images, la même cadence et la même taille que le modèle, avec la piste AAC d'origine copiée telle quelle.
- `4_final/final_9x16.mp4` (avec `--vertical`) : un recadrage qui suit la tête de Pierre, puis celle de Basil au moment où chacun parle, même quand Pierre tend le bras. Pour un recadrage fixe au centre, ajoute `--vertical-focus center`. Le recadrage fait 404x718 pixels, agrandi par défaut en 720x1280 (×1,78). Pour éviter cet agrandissement, utilise `--vertical-size 404x718`.

Le script vérifie automatiquement le nombre d'images, la cadence, la durée et la présence de l'audio.

## Tout enchaîner

Commande `all` : elle lance l'étape 1, puis s'arrête tant que `--pierre-image` et `--basil-image` ne sont pas fournis. Ajoute `--lipsync` pour inclure l'étape 3.

## Coût et durée estimés

Les prix viennent des pages fal.ai et sont à revérifier.

| Étape | Appels | Coût | Durée |
|---|---|---|---|
| 1. refs (3 variantes chacun, Nano Banana Pro 2K) | 2 | environ 0,90 $ | 1-2 min |
| 2. replace `wan-isolate` 720p (409 images / 16 × 0,08 $) | 2 | environ 4,10 $ | quelques minutes par passe, non vérifié (un autre hébergeur annonce environ 4 min) + environ 2 min en local |
| 2. variante `recast` 768P | 2 | environ 5,10 $ | non vérifié |
| 3. lipsync sync-3 (8 $/min) | 2 | environ 4,60 $ | non vérifié |
| 4. finish | 0 | 0 $ | environ 10 s |

Une itération complète coûte environ 5 $ sans lip-sync, et environ 10 $ avec. Prévois 2 ou 3 itérations.

## Points non vérifiés (à surveiller au premier vrai lancement)

- **Sorties de Wan sur fal :** on ne connaît ni la cadence de sortie, ni le nombre d'images, ni si l'audio est conservé. Le script recale la vidéo de toute façon, et le journal indique le mode utilisé (`frame-for-frame`, `resample` ou `stretch`).
- **Choix de la personne par fal :** on suppose qu'il prend la plus grande personne, comme le code open source de Wan. On ne sait pas si fal applique la même règle.
- **Masques de composition :** le masque de différence est une solution maison. Si Wan modifie trop le décor, passe `--diff-threshold` à 30-40, ou utilise `--matte box`.
- **Positions dans `reel1_tracks.json` :** les boîtes du corps viennent d'un détecteur automatique (YOLOv4, une image sur 6). Les images 384 et 390 de Pierre, que le détecteur ratait, ont été ajoutées à l'œil. Les boîtes de tête ont été placées à l'œil sur une planche quadrillée, puis vérifiées sur une trentaine d'images.
- **Contrôle par détecteur :** YOLOv4 remplace le détecteur de Wan (yolov10m). Sur les entrées actuelles, il ne trouve aucune image où le mauvais homme serait choisi, mais il trouve 4 images analysées de la passe Pierre (une image sur 3) où aucune personne n'est détectée. On ne sait pas ce que fait Wan dans ce cas.
- **`options.active_speaker_detection` de sync v3 :** on ne sait pas si les coordonnées sont en pixels de la vidéo envoyée, ni si `auto_detect: false` est le bon interrupteur. On ne connaît pas non plus la taille ni la cadence de sortie (le script recale de toute façon).
- **`minimax/h3-max/recast` :** l'attribution des références de gauche à droite vient de la page fal. On ne sait pas si le prompt peut la modifier, ni si le 3ᵉ homme reste intact.
- **Limites des modèles d'image :** nombre de références et `num_images` maximum (Nano Banana Pro). On ne sait pas non plus si Google refuse d'habiller une vraie personne. Dans ce cas, le script passe automatiquement au modèle suivant.
- **Consentement :** Pierre et Basil doivent être d'accord pour l'utilisation de leur visage (règles d'utilisation de fal et de Google).

# Hailo Vision API

Version : `1.2.6`

API HTTP minimale pour exécuter un modèle `.hef` sur un accélérateur Hailo.
L'inférence est strictement déléguée à HailoRT : si le runtime ou le Hailo
n'est pas disponible, l'application échoue au démarrage (aucun fallback CPU).

## Installation sur la machine Hailo

Installer d'abord le driver Hailo et le paquet `hailort` fourni pour la version
de Python utilisée, puis :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HEF_PATH=/opt/models/yolov8s.hef
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Le fichier HEF doit être compilé pour la puce installée (Hailo-8/8L/10H).

## API

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/v1/infer \
  -F image=@image.jpg
```

La réponse contient les sorties brutes nommées du modèle :

```json
{"outputs":{"yolov8_nms": [[...]]}}
```

Le décodage des boîtes et des classes dépend du HEF choisi ; il est volontairement
laissé à l'appelant afin que l'API reste compatible avec plusieurs modèles.

## Visualisation des deux caméras

```bash
CAMERA_USE_SUDO=1 .venv/bin/python -m uvicorn app.camera_server:app \
  --host 192.168.1.44 --port 8090
```

Ouvrir ensuite [http://192.168.1.44:8090](http://192.168.1.44:8090) depuis un
appareil du réseau local. La page affiche les flux MJPEG des caméras 0 et 1.
Le serveur n'écoute que sur l'adresse LAN du Pi.

La page affiche aussi la détection des personnes réalisée par le Hailo, avec
compteur, rectangles et alerte à l'apparition. Les résultats sont disponibles
sur `/detections/0` et `/detections/1`. Le modèle par défaut est
`/usr/local/hailo/resources/models/hailo8/yolov8m.hef` et peut être remplacé
avec `HEF_PATH`.

Le réglage par défaut de l’OV5647 est 1920×1080 à 5 FPS. L’IMX708 reste à
640×480 à 5 FPS afin de conserver les deux flux simultanément dans les limites
des buffers DMA. Le 1920×1080 reste sélectionnable pour un usage avec une seule
caméra. L’enrôlement
live est autorisé uniquement lorsqu’au moins une personne est détectée ; une
détection sans correspondance est affichée comme `Personne non identifiée`.

L’enrôlement live est disponible avec les boutons de la page. Une image JPEG
de référence est conservée localement dans `data/people/`; elle peut être
consultée via `/people/{id}/image` et le nom peut être modifié avec
`PATCH /people/{id}`.

## Limite importante

Le CPU reçoit la requête, décode JPEG/PNG et prépare le tenseur d'entrée. Il ne
réalise pas l'inférence. Pour supprimer aussi ce prétraitement CPU, il faut
fournir des tenseurs déjà préparés ou utiliser une pipeline caméra/GStreamer
avec prétraitement matériel.

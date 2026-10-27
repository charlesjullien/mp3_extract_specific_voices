# Extraction de voix

Application locale qui isole la voix d’une personne dans une vidéo YouTube ou un fichier audio. Elle tourne sur [http://127.0.0.1:8000](http://127.0.0.1:8000). L’audio est lu et écrit dans le dossier `data/` de ce projet.

## Ce que fait le programme

1. Vous collez un lien YouTube, ou vous déposez un MP3, un WAV ou un M4A.
2. pyannote repère qui parle, et quand.
3. Chaque voix est proposée avec un extrait d’environ 8 secondes.
4. Vous en choisissez une. ffmpeg assemble toutes ses prises de parole dans un MP3.
5. `PRECISE=false` dans `.env` convient aux interviews longues. `PRECISE=true` sert aux vidéos courtes : les répliques d’une à quelques secondes sont séparées, puis le programme recolle lui-même celles d’une même personne. Cet assemblage découpe plus de morceaux, il dure donc plus longtemps.
6. Une pause de moins d’une seconde reste dans le même extrait. En mode précis, seules les pauses de moins de 0,4 seconde le font. Au-delà, chaque réplique de la personne choisie reste un extrait, avec un jingle entre deux.
7. Une case à cocher ajoute, ou non, jusqu’à 5 secondes du son qui précède chaque extrait séparé.
8. Le bouton **Transcrire** lance Whisper sur ce MP3 final. Le texte s’affiche et se télécharge.

La diarisation utilise [pyannote.audio](https://github.com/pyannote/pyannote-audio) (`speaker-diarization-community-1`). La transcription utilise [faster-whisper](https://github.com/SYSTRAN/faster-whisper), une mise en œuvre locale des modèles Whisper. WhisperX sert à aligner des mots sur des locuteurs ; ce parcours transcrit seulement le fichier déjà assemblé.

Si deux personnes parlent en même temps, l’autre voix peut rester audible : le programme découpe les moments où la personne choisie est détectée, il ne sépare pas deux voix superposées.

Le nombre de personnes est facultatif. Le renseigner rend le regroupement plus fiable.

## Confidentialité

- Le serveur écoute seulement `127.0.0.1`.
- L’audio ne part pas vers une API de transcription.
- La télémétrie pyannote et Hugging Face est désactivée.
- Le jeton Hugging Face télécharge les poids du modèle de diarisation. Il ne sert pas à envoyer l’audio.
- Les seuls accès réseau sont le téléchargement YouTube (yt-dlp) et, la première fois, le téléchargement des modèles.
- Quand les deux modèles ont déjà été chargés une fois, le lancement suivant bloque les nouveaux accès Hugging Face, sauf si `.env` contient `HF_HUB_OFFLINE=0`.

## Prérequis

- Python 3.10 ou plus récent
- ffmpeg (`ffmpeg -version`)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Sans carte NVIDIA, cette commande télécharge quand même les bibliothèques CUDA. Dans ce cas, installez plutôt la variante CPU de PyTorch avec [uv](https://docs.astral.sh/uv/) :

```bash
uv venv .venv
uv pip install --python .venv/bin/python --torch-backend cpu -r requirements.txt
```

Avec un GPU NVIDIA, retirez `--torch-backend cpu` pour utiliser CUDA. Le programme choisit le GPU tout seul s’il est disponible.

Si `python3 -m venv` demande le paquet `python3-venv`, la même commande `uv venv` ci-dessus suffit.

### Jeton Hugging Face

Le modèle `pyannote/speaker-diarization-community-1` est open source et soumis à des conditions d’utilisation.

1. Créez un compte sur [huggingface.co](https://huggingface.co/join).
2. Acceptez les conditions sur [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
3. Créez un jeton en lecture sur [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
4. Placez-le dans `.env` : `HF_TOKEN=hf_...`

Les poids Whisper se téléchargent au premier clic sur **Transcrire**, sans jeton. Le modèle se choisit avec `WHISPER_MODEL` dans `.env` (`small` par défaut, `medium` si vous voulez une transcription plus juste). `CPU_PERCENT` fixe la part des cœurs utilisés sur processeur (`100` va le plus vite). Un GPU NVIDIA est utilisé s’il est présent.

## Lancement

```bash
source .venv/bin/activate
python run.py
```

Ouvrez [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Windows 11

Le dossier `.venv` créé sous Linux ne se réutilise pas. Sur Windows, recréez-le.

1. Installez Python 3.12 depuis [python.org](https://www.python.org/downloads/windows/) en cochant « Add python.exe to PATH ».
2. Installez ffmpeg, puis ouvrez un nouveau terminal :

```powershell
winget install Gyan.FFmpeg
```

3. Dans le dossier du projet :

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python run.py
```

Sans carte NVIDIA, les paquets PyTorch du fichier `requirements.txt` peuvent être très lourds. Dans ce cas, après l’activation du venv :

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Avec une carte NVIDIA, laissez l’installation normale : le programme passe sur le GPU tout seul. Le jeton Hugging Face se met dans `.env` comme sous Linux. La page est [http://127.0.0.1:8000](http://127.0.0.1:8000).

Les fichiers de travail restent dans `data/` jusqu’au bouton **Effacer ce travail**.

## Tests

```bash
python -m pytest
```

Les tests de timeline et d’assemblage ffmpeg ne téléchargent aucun modèle.

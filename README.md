# Vaucluse Hackathon Project

Ce dépôt regroupe des notebooks et scripts utilisés pour l’extraction, le nettoyage, et le géoréférencement de métadonnées et de cartes postales du Vaucluse dans le cadre du Hackathon 2026 de l'École nationale des chartes (ENC).

## Scripts et notebooks python 
dans `scripts_notebooks/`
- `scraping_urls.ipynb` : collecte d’URLs à aller scraper, un par carte postale.
- `scraping_metadata.ipynb` : scraping des métadonnées à partir de la liste d'URLs. 
- `identification_lieu-dit.ipynb` : préparation/filtrage de base de données (JSON/CSV) à implémenter dans la pipeline avec QWEN. Nettoyage réussi mais infructueux. 
- `georeferencement_osmnx.ipynb` : géocodage avec OSMNX (moins performant que nomitim).
- `georeferencement_5m_mathias_garnier.py` : script de géoréférencement réalisé par https://github.com/MathiasGarnier.
- `script_cartes_postales.py` : pipeline de traitement OCR/transcription avec QWEN3 VL, identification communes, lieux-dits, monuments avec recherche web.
- `corpus_benchmark.ipynb` : à partir des 84 cartes traitées avec QWEN3 VL, on a défini une vérité de terrain manuellement pour benchmarker notre baseline.
- `Géoréférencement_nomitim_carto.ipynb` : notebook **prévu pour Google Colab** pour géolocaliser les monuments, lieux-dits puis créer la carte en html.

## Données

Les fichiers de données se trouvent dans `data/` (JSON, CSV, images, sorties intermédiaires).
- `metadata_web_complet.json` : résultat du scraping des métadonnées sur le site des archives du Vaucluse.
- dans `précision_geoguessr` les résultats obtenus par https://github.com/MathiasGarnier avec son script `georeferencement_5m_mathias_garnier.py`.

## Dépendances

Les dépendances sont listées dans `requirements.txt` et `requirements-min.txt` pour tous les notebooks à l'exception `Géoréférencement_nomitim_carto.ipynb` qui est un google colab. Le fichier `georeferencement_5m_mathias_garnier.py` a été réalisé par Mathias Garnier et les dépendances ne sont pas dans `requirements.txt` ni `requirements-min.txt`. Quant à `script_cartes_postales.py`, les dépendances sont en commentaire au début du script. 

## Notes

- Certains notebooks ont été exécutés dans des environnements différents (local vs Colab).
- Les chemins et sources de données peuvent nécessiter une adaptation locale.

## Participants 

Garnier Mathias : étudiant en M1 Humanités Numériques à l'ENC.
Létoffé Maxime : étudiant en M1 Humanités Numériques à l'ENC.
Rivière Mathieu : étudiant en M1 Humanités Numériques à l'ENC.
Vidal-Gorène Chahan : respondable du master Humanités Numériques à l'ENC.

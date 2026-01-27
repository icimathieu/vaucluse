# Vaucluse Hackathon Project

Ce dépôt regroupe des notebooks et scripts utilisés pour l’extraction, le nettoyage, et le géoréférencement de métadonnées et de cartes postales du Vaucluse dans le cadre du Hackathon 2026 de l'École nationale des chartes (ENC).

## Scripts et notebooks python 
dans `scripts_notebooks/`
- `scraping_urls.ipynb` : collecte d’URLs à aller scraper, un par carte postale.
- `scraping_metadata.ipynb` : scraping des métadonnées à partir de la liste d'URLs. 
- `identification_lieu-dit.ipynb` : préparation/filtrage de base de données (JSON/CSV) à implémenter dans la pipeline avec QWEN. Nettoyage réussi mais infructueux. 
- `georeferencement_osmnx.ipynb` : géocodage avec OSMNX (moins performant que nomitim).
- `georeferencement_5m_mathias_garnier.py` : script de géoréférencement réalisé par https://github.com/MathiasGarnier.
- `script_cartes_postales.py` : pipeline de traitement OCR/transcription avec QWEN3 VL, identification communes, lieux-dits, monuments avec recherche web. Script dérivé du Google colab : `etapes1-2_QWEN3.ipynb`.
- `corpus_benchmark.ipynb` : à partir des 84 cartes traitées avec QWEN3 VL, on a défini une vérité de terrain manuellement pour benchmarker notre baseline.
- `Géoréférencement_nominatim_carto.ipynb` : notebook **prévu pour Google Colab** pour géolocaliser les monuments, lieux-dits puis créer la carte en html.

## Données

Les fichiers de données se trouvent dans `data/` (JSON, CSV, images, sorties intermédiaires).
- dans `métadonnées` : `metadata_output_merged.json` : résultat du scraping des métadonnées sur le site des archives du Vaucluse et output de la pipeline avec QWEN3 VL. `vaucluse_georef_nominatim4.csv` : le même fichier auquel s'ajoute les coordonnées des monuments, lieux-dits si trouvés par nominatim (avec condition de distance pour qu'il n'y ait pas de données aberrantes). 
- dans `précision_geoguessr` les résultats obtenus par https://github.com/MathiasGarnier avec son script `georeferencement_5m_mathias_garnier.py`.
- `carte_vaucluse_interactive5.html` : la carte interactive avec popup et métadonnées des cartes postales (images comprises!), placées en fonction du meilleur degré de précision.
- dans `benchmark_84cartes` les json obtenus et une note .txt importante sur nos résultats au global.

## Dépendances

Les dépendances sont listées dans `requirements.txt` et `requirements-min.txt` pour tous les notebooks à l'exception `Géoréférencement_nomitim_carto.ipynb` et `etapes1-2_QWEN3.ipynb` qui sont des google colab. Le fichier `georeferencement_5m_mathias_garnier.py` a été réalisé par Mathias Garnier et les dépendances ne sont pas dans `requirements.txt` ni `requirements-min.txt`. Quant à `script_cartes_postales.py`, les dépendances sont en commentaire au début du script.

## Notes

- Certains notebooks ont été exécutés dans des environnements différents (local vs Colab).
- Les chemins et sources de données peuvent nécessiter une adaptation locale.

## Participants 

Garnier Mathias : étudiant en M1 Humanités Numériques à l'ENC.
Létoffé Maxime : étudiant en M1 Humanités Numériques à l'ENC.
Rivière Mathieu : étudiant en M1 Humanités Numériques à l'ENC.
Vidal-Gorène Chahan : respondable du master Humanités Numériques à l'ENC.

## Partenaires

| École nationale des chartes - PSL | Département du Vaucluse |
| --- | --- |
| ![École des chartes](data/logos/logo-chartes-psl-coul_0.png) | ![Vaucluse](data/logos/logo-dept-vaucluse-2025.svg_.png) |

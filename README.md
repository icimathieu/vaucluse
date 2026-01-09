# Vaucluse Hackathon Project

Ce dépôt regroupe des notebooks et scripts utilisés pour l’extraction, le nettoyage, et le géoréférencement de métadonnées et de cartes postales du Vaucluse.

## Contenu principal

- `scraping_urls.ipynb` : collecte d’URLs.
- `scraping_metadata.ipynb` : scraping des métadonnées.
- `identification_lieu-dit.ipynb` : préparation/filtrage de référentiels (JSON/CSV).
- `georeferencement_osmnx.ipynb` : géocodage avec OSMNX.
- `georeferencement_5m_mathias_garnier.py` : script de géoréférencement.
- `script_cartes_postales.py` : pipeline de traitement OCR/transcription.
- `corpus_benchmark.ipynb` : **réalisé par Mathias Garnier**.
- `Géoréférencement_nomitim_carto.ipynb` : notebook **prévu pour Google Colab**.

## Données

Les fichiers de données se trouvent dans `data/` (JSON, CSV, images, sorties intermédiaires).

## Dépendances

Les dépendances sont listées dans `requirements.txt` et `requirements-min.txt` pour tous les notebooks à l'exception `Géoréférencement_nomitim_carto.ipynb` qui est un google colab. Le fichier `georeferencement_5m_mathias_garnier.py` a été réalisé par Mathias Garnier et les dépendances ne sont pas dans `requirements.txt` ni `requirements-min.txt`. Quant à `script_cartes_postales.py`, les dépendances sont en commentaire au début du script. 

## Notes

- Certains notebooks ont été exécutés dans des environnements différents (local vs Colab).
- Les chemins et sources de données peuvent nécessiter une adaptation locale.

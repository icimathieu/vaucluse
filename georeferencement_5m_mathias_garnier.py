import osmnx as ox
import pandas as pd

# 1. Forcer l'absence de cache pour éviter de recharger une vieille requête erronée
ox.settings.use_cache = False

def verifier_et_recuperer_cibles(lieu_precis):
    print(f"--- Recherche pour : {lieu_precis} ---")
    
    # Tags pour les points d'intérêts (églises, lieux historiques...)
    tags = {
        'building': ['church', 'cathedral', 'chapel', 'civic'], # ajouter des trucs
        'historic': True,
        'tourism': 'attraction',
        'man_made': 'tower',
        'bridge': True
    }
    
    try:
        # Récupération
        gdf = ox.features_from_place(lieu_precis, tags=tags)
        
        # 2. VÉRIFICATION CRITIQUE : Forcer la projection en Lat/Lon (EPSG:4326)
        # C'est ce qui garantit que vous lisez des coordonnées GPS standards
        gdf = gdf.to_crs("epsg:4326")
        
        print(f"Nombre de lieux trouvés : {len(gdf)}")
        
        return gdf

    except Exception as e:
        print(f"Erreur : {e}")
        return None

cibles = verifier_et_recuperer_cibles("Avignon, 84007, France")

# Afficher les 3 premiers résultats avec leur nom pour vérifier
if cibles is not None and not cibles.empty:
    # On gère le cas où le nom n'est pas renseigné
    if 'name' in cibles.columns:
        print("\nExemples de lieux trouvés :")
        print(cibles['name'].dropna().head(3))
######################################################
####### Installation
######################################################

#pip install unsloth xformers bitsandbytes accelerate sentencepiece protobuf datasets huggingface_hub hf_transfer torch Pillow -U duckduckgo_search

import os, re
from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 vision support
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # Can fit in a 80GB card!
    "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

    "unsloth/Pixtral-12B-2409-bnb-4bit",              # Pixtral fits in 16GB!
    "unsloth/Pixtral-12B-Base-2409-bnb-4bit",         # Pixtral base model

    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",          # Qwen2 VL support
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",

    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",      # Any Llava variant works!
    "unsloth/llava-1.5-7b-hf-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

######################################################
####### ETAPE 1 : OCR ET INDEXATION PRELIMINAIRE AVEC QWEN
######################################################

import os
import json
from transformers import TextStreamer

# 1. Config
dataset_path = './data'
output_json_path = './transcriptions_classe1.json' # Chemin de sortie
extensions = (".jpg", ".jpeg", ".png")

image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(extensions)]

# 2. Préparation Modèle
FastVisionModel.for_inference(model)
# On retire le streamer pour plus de clarté pendant l'enregistrement,
# ou on le garde juste pour voir passer le texte.

all_results = [] # Liste pour stocker les données

print(f"Traitement de {len(image_files)} images...")

# 3. Boucle avec extraction
for image_name in image_files:
    image_path = os.path.join(dataset_path, image_name)
    print(f"Processing: {image_name}")

    instruction = "Act as an OCR and location classifier. Extract text from the image. Output ONLY a JSON object. Keys: 'raw_text', 'city', 'hamlet', 'monument'. If a field is empty, use null. For the location classification, ignore the name Vaucluse. If there is a mention of a toponym 'près de' (near another toponym), always chose the first toponym mentionned because it is the most precise location indicator. For the location informations, don't put the articles like 'L', 'Le' or 'La'. Generate just one dictionnary for each file, don't put a dictionnary inside another dictionnary. Each city should be a toponym in this list : Ansouis, Apt, Aurel, Avignon, Barroux (Le), Bastide-des-Jourdans (La), Bastidonne (La), Beaucet (Le), Beaumes-de-Venise, Beaumettes, Beaumont-de-Pertuis, Bédarrides, Bedoin, Blauvac, Bollène, Bonnieux, Brantes, Buoux, Cabrières-d'Aigues, Cadenet, Caderousse, Camaret-sur-Aigues, Caromb, Carpentras, Caumont-sur-Durance, Cavaillon, Châteauneuf-du-Pape, Courthézon, Crillon-le-Brave, Cucuron, Entraigues-sur-la-Sorgue, Fontaine-de-Vaucluse, Gargas, Gigondas, Gordes, Goult, Grambois, Grillon, Isle-sur-la-Sorgue (L'), Jonquières, Joucas, Lacoste, Lapalud, Lioux, Lourmarin, Malaucène, Mazan, Ménerbes, Mérindol, Mirabeau, Mondragon, Monieux, Monteux, Morières-lès-Avignon, Mormoiron, Mornas, Murs, Orange, Pernes-les-Fontaines, Pertuis, Peypin-d'Aigues, Piolenc, Pontet (Le), Puyvert, Rasteau, Richerenches, Rustrel, Sablet, Saignon, Sainte-Cécile-les-Vignes, Saint-Christol, Saint-Didier, Saint-Martin-de-Castillon, Saint-Martin-de-la-Brasque, Saint-Pantaléon, Saint-Saturnin-lès-Apt, Sault, Saumanes-de-Vaucluse, Savoillans, Sérignan-du-Comtat, Sorgues, Taillades, Thor (Le), Tour-d'Aigues (La), Vacqueyras, Vaison-la-Romaine, Valréas, Vaugines, Venasque, Viens, Villars, Villes-sur-Auzon, Visan, Mont Ventoux, Dentelles de Montmirail. You can only put element of this list in 'city'. If you see an element of this list of cities, don't put it in 'hamlet'. Then, if you see a mention of a 'Hameau', 'Quartier' or a specific place name that is not the city, put it in the 'hamlet' field. Be very concise (e.g., 'Sainte-Colombe' instead of 'Hameau de Sainte-Colombe').  "

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    inputs = tokenizer(image_path, input_text, add_special_tokens = False, return_tensors = "pt").to("cuda")

    # Génération sans streamer pour récupérer proprement la string
    output_ids = model.generate(
        **inputs,
        max_new_tokens = 256,
        use_cache = True,
        temperature = 0.1
    )

    # Décodage de la réponse
    full_response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    # Nettoyage de la réponse pour ne garder que le texte après l'instruction
    # (Dépend de comment votre tokenizer décode, souvent on sépare par le prompt de réponse)
    response_text = full_response.split("assistant\n")[-1] if "assistant\n" in full_response else full_response

    # On ajoute le résultat à notre liste
    result_entry = {
        "file_name": image_name,
        "raw_output": response_text.strip()
    }
    all_results.append(result_entry)

# 4. Sauvegarde 
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)

print(f"\nFichier JSON sauvegardé ici : {output_json_path}")

######################################################
####### ETAPE 2 : INDEXATION DANS 3 CLASSES GEO AVEC QWEN ET RECHERCHE WEB
######################################################



import os
import json
import re
import warnings
from duckduckgo_search import DDGS
from PIL import Image

# 1. Configuration

input_json_path = './transcriptions_classe1.json'
output_path = './transcriptions_coherentes_final.json'
images_dir = './data'

# Mémoires globales (Cache)
memoire_monuments = {}
memoire_hameaux = {}

# --- Fonctions de Nettoyage et Recherche ---

def normalize_text(text):
    """Nettoie le texte : minuscule, sans articles au début, sans ponctuation"""
    if not text or text.lower() in ["inconnu", "null", "none"]: return ""
    t = text.lower().strip()

    # Retrait des articles au début (ex: Le Portalet -> portalet)
    # Gestion des articles simples
    t = re.sub(r"^(le|la|les|un|une|au|aux|du|des|de)\s+", "", t)
    # Gestion des élisions (L', D') avec ou sans espace
    t = re.sub(r"^(l'|d')\s*", "", t)

    # Retrait ponctuation
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())

def run_qwen(image_path, instruction):
    try:
        raw_image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(raw_image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=200, use_cache=True, temperature=0.1)
        res = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        return res.split("assistant\n")[-1].strip()
    except:
        return "{}"

# 2. Chargement du JSON source
with open(input_json_path, 'r', encoding='utf-8') as f:
    data_sources = json.load(f)

final_results = []
FastVisionModel.for_inference(model)

# 3. Boucle de Traitement
for i, entry in enumerate(data_sources):
    fname = entry['file_name']
    path = os.path.join(images_dir, fname)
    if not os.path.exists(path): path = os.path.join(images_dir, fname.replace('copy_', ''))
    if not os.path.exists(path): continue

    # Extraction infos
    raw_data = json.loads(entry['raw_output'])
    text_ocr = raw_data.get('raw_text', '')
    city = raw_data.get('city', '')
    monument_precedent = raw_data.get('monument', None)
    hamlet_precedent = raw_data.get('hamlet', None)

    print(f"\n[{i+1}/{len(data_sources)}] Analyse : {os.path.basename(path)}")

    # --- ÉTAPE 1 : IDENTIFICATION ---
    prompt_id = f"""
    Analyse ce texte : "{text_ocr}"
    Ville actuelle : {city}
    Suggestions précédentes : Monument="{monument_precedent}", Lieu-dit="{hamlet_precedent}".

    CONSIGNE CRUCIALE :
    - Le champ 'hamlet' doit être un quartier, un hameau ou un lieu-dit spécifique.
    - NE METS PAS le nom de la ville ou une version courte de la ville dans 'hamlet' (ex: si ville='Vaison-la-Romaine', 'Vaison' est INTERDIT).
    - Tu ne peux dans aucun cas modifier le champ 'city'.
    - Si aucun quartier/lieu-dit n'est nommé, mets "Inconnu".
    - Dans hamlet, tu ne peux en aucun cas mettre ces éléments : Ansouis, Apt, Aurel, Avignon, Le Barroux, La Bastide-des-Jourdans, La Bastidonne, Le Beaucet, Beaumes-de-Venise, Beaumettes, Beaumont-de-Pertuis, Bédarrides, Bedoin, Blauvac, Bollène, Bonnieux, Brantes, Buoux, Cabrières-d'Aigues, Cadenet, Caderousse, Camaret-sur-Aigues, Caromb, Carpentras, Caumont-sur-Durance, Cavaillon, Châteauneuf-du-Pape, Courthézon, Crillon-le-Brave, Cucuron, Entraigues-sur-la-Sorgue, Fontaine-de-Vaucluse, Gargas, Gigondas, Gordes, Goult, Grambois, Grillon, L'Isle-sur-la-Sorgue, Isle-sur-la-Sorgue, Jonquières, Joucas, Lacoste, Lapalud, Lioux, Lourmarin, Malaucène, Mazan, Ménerbes, Mérindol, Mirabeau, Mondragon, Monieux, Monteux, Morières-lès-Avignon, Mormoiron, Mornas, Murs, Orange, Pernes-les-Fontaines, Pertuis, Peypin-d'Aigues, Piolenc, Le Pontet, Rasteau, Richerenches, Rustrel, Sablet, Saignon, Sainte-Cécile-les-Vignes, Saint-Christol, Saint-Didier, Saint-Martin-de-Castillon, Saint-Martin-de-la-Brasque, Saint-Pantaléon, Saint-Saturnin-lès-Apt, Sault, Saumanes-de-Vaucluse, Savoillans, Sérignan-du-Comtat, Sorgues, Taillades, Le Thor, La Tour-d'Aigues, Vacqueyras, Vaison-la-Romaine, Valréas, Vaugines, Venasque, Viens, Villars, Villes-sur-Auzon, Visan, Ventoux, Dentelles de Montmirail.
    - S'il y a uniquement la mention "église" et pas d'autre précision dans le texte des cartes, renvoie simplement Eglise de {city} en monument.
    Réponds en JSON :
    {{"monument": "...", "monument_trouve": true/false, "hamlet": "...", "hamlet_trouve": true/false, "confidence": 0.X}}
    """

    res_id = run_qwen(path, prompt_id)
    try:
        info = json.loads(re.search(r'\{.*\}', res_id, re.DOTALL).group(0))
    except:
        info = {"monument_trouve": False, "hamlet_trouve": False, "confidence": 0}

    # --- ÉTAPE 2 : CACHE MONUMENT ---
    sujet_m = info.get('monument', 'Inconnu')
    m_norm = normalize_text(sujet_m)
    cache_key_m = (city, m_norm)

    if not info.get('monument_trouve') or m_norm == "":
        monument_final = "Aucun monument"
    elif cache_key_m in memoire_monuments:
        monument_final = memoire_monuments[cache_key_m]
    else:
        monument_final = sujet_m
        memoire_monuments[cache_key_m] = monument_final
        print(f"💾 CACHE MONUMENT : {monument_final}")

    # --- ÉTAPE 3 : VALIDATION TOPO ---
    sujet_h = info.get('hamlet', 'Inconnu')
    h_norm = normalize_text(sujet_h)
    cache_key_h = (city, h_norm)

    hamlet_final = "Aucun lieu-dit"
    topo_match = None

    if info.get('hamlet_trouve') and h_norm != "":
        if cache_key_h in memoire_hameaux:
            hamlet_final, topo_match = memoire_hameaux[cache_key_h]
        else:
            # Recherche dans la base topographique (avec recherche par mot classant)
            hamlet_final = sujet_h
            memoire_hameaux[cache_key_h] = (hamlet_final, topo_match)


    # --- COMPILATION ---
    final_results.append({
        "file_name": fname,
        "city": city,
        "monument_uniformise": monument_final,
        "hamlet_uniformise": hamlet_final,
        "confidence": info.get('confidence', 0),
        "ocr_text": text_ocr
    })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

print(f"\n✅ Analyse terminée. Résultats dans : {output_path}")
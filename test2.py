import os
import pandas as pd
import json
import base64
from mistralai import Mistral
from dotenv import load_dotenv
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pyfiglet import Figlet
from termcolor import colored
import subprocess

# Chargement des variables d'environnement
load_dotenv()
api_key = os.environ.get("MISTRAL_API_KEY")

client = Mistral(api_key=api_key)

# Configuration
MATCHING_RULES = {
    'montant_tolerance': 0.01,
    'jours_tolerance': 3,
    'seuil_similarite': 0.7
}

def display_image_terminal(image_path):
    """Affiche un aperçu de la facture en terminal"""
    try:
        # Solution 1: ASCII Art (nécessite img2txt ou libcaca)
        try:
            subprocess.run(["img2txt", image_path], check=True)
            return
        except:
            pass
        
        # Solution 2: Affichage stylisé du nom
        f = Figlet(font='small')
        print(colored(f.renderText('FACTURE'), 'cyan'))
        print(colored(f"Fichier: {os.path.basename(image_path)}", 'yellow'))
        print(colored("="*40, 'blue'))
        
        # Solution 3: Aperçu base64 (très basique)
        with open(image_path, "rb") as f:
            print(colored(f"Aperçu base64 (début): {base64.b64encode(f.read())[:30]}...", 'dark_grey'))
            
    except Exception as e:
        print(colored(f"Impossible d'afficher l'aperçu: {e}", 'red'))

def parse_date(date_str):
    """Parse les dates avec plusieurs formats"""
    formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%Y%m%d']
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None

# [Les autres fonctions (calculate_vendor_similarity, find_matches, etc.) restent identiques]

def process_invoices(image_folder):
    """Traitement des images de factures avec aperçu terminal"""
    invoices = []
    model = "pixtral-12b-2409"
    
    system_message = {
        "role": "system",
        "content": (
            "Extract EXACT JSON from invoices with these keys:\n"
            "{\n"
            "  \"date\": \"YYYY-MM-DD\",\n"
            "  \"amount\": float,\n"
            "  \"currency\": \"XXX\",\n"
            "  \"vendor\": \"string\",\n"
            "  \"invoice_number\": \"string\"\n"
            "}\n"
            "NO other text or explanations."
        )
    }

    for img_file in sorted(os.listdir(image_folder)):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(image_folder, img_file)
        
        # Aperçu dans le terminal
        print("\n" + colored("="*60, 'blue'))
        print(colored(f"TRAITEMENT DE LA FACTURE: {img_file}", 'white', 'on_blue'))
        display_image_terminal(img_path)
        print(colored("="*60, 'blue') + "\n")

        try:
            with open(img_path, "rb") as f:
                base64_img = base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(colored(f"Erreur lecture {img_file}: {e}", 'red'))
            continue

        try:
            response = client.chat.complete(
                model=model,
                messages=[
                    system_message,
                    {"role": "user", "content": [
                        {"type": "text", "text": "Extract data"},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_img}"}
                    ]}
                ],
                response_format={"type": "json_object"}
            )
            
            data = json.loads(response.choices[0].message.content)
            data['amount'] = float(data['amount'])
            data['currency'] = data['currency'].upper()
            data['date_parsed'] = parse_date(data['date'])
            
            if not data['date_parsed']:
                raise ValueError(f"Date invalide: {data['date']}")
                
            data['source_file'] = img_file
            invoices.append(data)
            
            print(colored("DONNÉES EXTRAITES:", 'green'))
            print(f"- Date: {colored(data['date'], 'yellow')}")
            print(f"- Montant: {colored(data['amount'], 'yellow')} {colored(data['currency'], 'yellow')}")
            print(f"- Fournisseur: {colored(data['vendor'], 'yellow')}")
            
        except Exception as e:
            print(colored(f"Erreur traitement {img_file}: {e}", 'red'))
            continue
    
    return invoices

# [Les autres fonctions (process_transactions, generate_report, main) restent identiques]

if __name__ == "__main__":
    # Installation des dépendances nécessaires pour l'affichage terminal
    try:
        import pyfiglet
        import termcolor
    except ImportError:
        print("Installation des dépendances pour l'affichage terminal...")
        subprocess.run(["pip", "install", "pyfiglet", "termcolor"], check=True)
    
    main()
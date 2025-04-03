import streamlit as st
import pandas as pd
import os
import tempfile
from PIL import Image
import io
import base64
import re
import json
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
from mistralai import Mistral
from dotenv import load_dotenv
import shutil


load_dotenv()
api_key = os.environ.get("MISTRAL_API_KEY")

client = Mistral(api_key=api_key)

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Correspondance Factures-Relevés",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Style CSS pour une interface épurée
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        padding: 0.5rem;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .upload-box {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .results-box {
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 1.5rem;
        background-color: #f9f9f9;
        margin-top: 2rem;
    }
    .stats-box {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .facture-link {
        color: #1E88E5;
        text-decoration: underline;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration du modèle (importé du code fourni)
MATCHING_RULES = {
    'montant_tolerance': 0.01,
    'jours_tolerance': 3,
    'seuil_similarite': 0.7
}

def parse_date(date_str):
    """Parse les dates avec plusieurs formats"""
    formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%Y%m%d']
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None

def calculate_vendor_similarity(vendor1, vendor2):
    """Calcule la similarité cosinus entre deux noms de vendeurs"""
    vectorizer = TfidfVectorizer()
    try:
        vectors = vectorizer.fit_transform([vendor1, vendor2])
        return cosine_similarity(vectors[0], vectors[1])[0][0]
    except:
        return 0.0

def find_matches(invoice, transactions):
    """Matching en cascade selon les règles métiers"""
    # Étape 1: Filtre strict par montant
    matches = [t for t in transactions
              if abs(t['amount'] - invoice['amount']) <= MATCHING_RULES['montant_tolerance']]

    if len(matches) == 1:
        return matches[0], 1.0, {'method': 'montant_unique'}

    # Étape 2: Filtre par date si plusieurs montants identiques
    if len(matches) > 1:
        date_matches = [t for t in matches
                       if abs((t['date_parsed'] - invoice['date_parsed']).days) <= MATCHING_RULES['jours_tolerance']]

        if len(date_matches) == 1:
            return date_matches[0], 1.0, {'method': 'date_unique'}

        # Étape 3: Similarité du vendeur si nécessaire
        if len(date_matches) > 1:
            best_match = None
            best_score = 0
            for t in date_matches:
                score = calculate_vendor_similarity(invoice['vendor'], t['vendor'])
                if score > best_score:
                    best_score = score
                    best_match = t

            if best_score >= MATCHING_RULES['seuil_similarite']:
                return best_match, best_score, {'method': 'similarite_vendeur'}

    # Aucun match trouvé
    return None, 0.0, {'method': 'non_trouve'}

def process_invoices(image_folder):
    """Traitement des images de factures"""
    invoices = []

    # Récupération de l'API key du modèle Mistral
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        st.error("Clé API Mistral non trouvée. Veuillez configurer la variable d'environnement MISTRAL_API_KEY.")
        return []

    client = Mistral(api_key=api_key)
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

    for img_file in os.listdir(image_folder):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(image_folder, img_file)

        try:
            with open(img_path, "rb") as f:
                base64_img = base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            st.error(f"Erreur lecture {img_file}: {e}")
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

        except Exception as e:
            st.error(f"Erreur traitement {img_file}: {e}")
            continue

    return invoices

def process_transactions(csv_folder):
    """Traitement des relevés bancaires CSV"""
    transactions = []

    for csv_file in os.listdir(csv_folder):
        if not csv_file.lower().endswith('.csv'):
            continue

        csv_path = os.path.join(csv_folder, csv_file)

        try:
            # Essayer plusieurs encodages
            encodings = ['utf-8', 'latin-1', 'windows-1252']
            df = None
            for enc in encodings:
                try:
                    df = pd.read_csv(csv_path, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                st.error(f"Erreur: Impossible de lire {csv_file} avec les encodages testés")
                continue

            # Normalisation des colonnes
            df.columns = df.columns.str.lower().str.strip()

            # Détection des colonnes
            col_map = {
                'date': ['date', 'transaction date', 'date operation'],
                'amount': ['amount', 'montant', 'debit', 'credit'],
                'vendor': ['vendor', 'description', 'nom', 'fournisseur'],
                'currency': ['currency', 'devise']
            }

            cols = {}
            for field, alternatives in col_map.items():
                for alt in alternatives:
                    if alt in df.columns:
                        cols[field] = alt
                        break

            if not all(cols.get(k) for k in ['date', 'amount', 'vendor']):
                st.error(f"Colonnes manquantes dans {csv_file}")
                continue

            # Traitement ligne par ligne
            for idx, row in df.iterrows():
                try:
                    amount = abs(float(row[cols['amount']]))
                    date_parsed = parse_date(str(row[cols['date']]))

                    if not date_parsed:
                        continue

                    transactions.append({
                        'line_number': idx + 2,  # +2 car idx commence à 0 et la ligne 1 est l'en-tête
                        'date': str(row[cols['date']]),
                        'date_parsed': date_parsed,
                        'amount': amount,
                        'currency': (row[cols['currency']] if cols.get('currency') else 'EUR').upper(),
                        'vendor': str(row[cols['vendor']]).strip(),
                        'source_file': csv_file,
                        'raw_data': row.to_dict()
                    })
                except Exception as e:
                    st.error(f"Erreur ligne {idx+2} dans {csv_file}: {e}")

        except Exception as e:
            st.error(f"Erreur fichier {csv_file}: {e}")

    return transactions

def generate_report(results):
    """Génération du rapport adapté"""
    # Création du rapport selon le format attendu par l'interface Streamlit
    matches = []
    unmatched = []

    stats = {
        'total_invoices': len(results),
        'matched_strict': 0,
        'matched_date': 0,
        'matched_vendor': 0,
        'unmatched': 0
    }

    for res in results:
        if res['transaction']:
            match_entry = {
                'Facture': res['invoice']['source_file'],
                'Relevé bancaire': res['transaction']['source_file'],
                'Détail transaction': (f"Transaction du {res['transaction']['date']} - "
                                    f"{res['transaction']['vendor']} - "
                                    f"{res['transaction']['amount']} {res['transaction']['currency']}"),
                'Ligne transaction': f"Ligne {res['transaction']['line_number']}"  # Ajout du numéro de ligne
            }
            matches.append(match_entry)

            # Mise à jour des stats
            if res['score'] == 1.0:
                if res['details']['method'] == 'montant_unique':
                    stats['matched_strict'] += 1
                else:
                    stats['matched_date'] += 1
            elif res['score'] >= MATCHING_RULES['seuil_similarite']:
                stats['matched_vendor'] += 1
        else:
            unmatched.append({'Facture': res['invoice']['source_file']})
            stats['unmatched'] += 1

    return matches, unmatched, stats

# Fonction pour créer un lien de téléchargement
def get_download_link(df, filename, text):
    """Génère un lien pour télécharger le dataframe en format Excel"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')

    b64 = base64.b64encode(output.getvalue()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Fonction pour afficher une image depuis les données encodées
def display_invoice_image(img_data):
    try:
        image = Image.open(io.BytesIO(img_data))
        return image
    except Exception as e:
        st.error(f"Erreur d'affichage de l'image: {e}")
        return None

# Initialisation des variables de session
if 'facture_images' not in st.session_state:
    st.session_state['facture_images'] = {}
if 'selected_invoice' not in st.session_state:
    st.session_state['selected_invoice'] = None
if 'processed' not in st.session_state:
    st.session_state['processed'] = False

# Fonction pour gérer les clics sur les factures
def view_invoice(filename):
    st.session_state['selected_invoice'] = filename
    st.rerun()  # Remplace st.experimental_rerun() par st.rerun()

# Titre de l'application
st.title("Correspondance Automatique Factures / Relevés Bancaires")

# Si une facture est sélectionnée, l'afficher puis retourner au tableau principal
if st.session_state['selected_invoice'] is not None:
    invoice_filename = st.session_state['selected_invoice']
    if invoice_filename in st.session_state['facture_images']:
        st.subheader(f"Facture: {invoice_filename}")
        img_data = st.session_state['facture_images'][invoice_filename]
        image = display_invoice_image(img_data)
        if image:
            st.image(image, caption=invoice_filename)
        else:
            st.error("Impossible d'afficher cette facture.")

    if st.button("Retour au tableau"):
        st.session_state['selected_invoice'] = None
        st.rerun()  # Remplace st.experimental_rerun() par st.rerun()
else:
    # Interface principale
    # Création de deux colonnes pour l'interface
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        st.subheader("Importation des relevés bancaires (CSV)")
        bank_files = st.file_uploader("Glissez ou sélectionnez vos fichiers CSV",
                                    type=['csv'],
                                    accept_multiple_files=True,
                                    help="Formats acceptés: CSV")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        st.subheader("Importation des factures (JPG)")
        facture_files = st.file_uploader("Glissez ou sélectionnez vos factures",
                                        type=['jpg', 'jpeg', 'png'],
                                        accept_multiple_files=True,
                                        help="Formats acceptés: JPG, JPEG, PNG")
        st.markdown('</div>', unsafe_allow_html=True)

    # Bouton pour lancer l'analyse
    if st.button("ANALYSER ET TROUVER LES CORRESPONDANCES"):
        if bank_files and facture_files:
            with st.spinner("Traitement en cours..."):
                # Création de dossiers temporaires pour stocker les fichiers importés
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Création des sous-dossiers
                    bank_dir = os.path.join(temp_dir, "bank_statements")
                    images_dir = os.path.join(temp_dir, "images")
                    os.makedirs(bank_dir, exist_ok=True)
                    os.makedirs(images_dir, exist_ok=True)

                    # Sauvegarde des relevés bancaires
                    for bank_file in bank_files:
                        file_path = os.path.join(bank_dir, bank_file.name)
                        with open(file_path, "wb") as f:
                            f.write(bank_file.getbuffer())

                    # Sauvegarde des factures et stockage des images en mémoire
                    st.session_state['facture_images'] = {}
                    for facture_file in facture_files:
                        # Stockage de l'image en mémoire pour l'affichage ultérieur
                        st.session_state['facture_images'][facture_file.name] = facture_file.getvalue()

                        # Sauvegarde sur disque pour le traitement
                        file_path = os.path.join(images_dir, facture_file.name)
                        with open(file_path, "wb") as f:
                            f.write(facture_file.getbuffer())

                    # Traitement avec le modèle existant
                    invoices = process_invoices(images_dir)
                    if not invoices:
                        st.error("Aucune facture valide n'a pu être traitée. Vérifiez les fichiers images et l'API Mistral.")
                    else:
                        transactions = process_transactions(bank_dir)
                        if not transactions:
                            st.error("Aucune transaction valide n'a pu être traitée. Vérifiez les fichiers CSV.")
                        else:
                            # Matching
                            results = []
                            for invoice in invoices:
                                match, score, details = find_matches(invoice, transactions)
                                results.append({
                                    'invoice': invoice,
                                    'transaction': match,
                                    'score': score,
                                    'details': details
                                })

                            # Génération du rapport
                            matches, unmatched, stats = generate_report(results)

                            # Enregistrement des résultats dans la session
                            st.session_state['matches'] = matches
                            st.session_state['unmatched'] = unmatched
                            st.session_state['stats'] = stats
                            st.session_state['processed'] = True

                st.success("Analyse terminée!")
        else:
            st.warning("Veuillez importer au moins un relevé bancaire et une facture.")

    # Affichage des résultats si le traitement a été effectué
    if st.session_state['processed']:
        matches = st.session_state['matches']
        unmatched = st.session_state['unmatched']
        stats = st.session_state['stats']

        st.markdown('<div class="results-box">', unsafe_allow_html=True)

        # Statistiques
        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Factures traitées", stats['total_invoices'])
        with col2:
            st.metric("Factures correspondantes", stats['total_invoices'] - stats['unmatched'])
        with col3:
            st.metric("Factures non correspondantes", stats['unmatched'])
        st.markdown('</div>', unsafe_allow_html=True)

        # Tableau des correspondances avec les liens vers les factures
        if matches:
            st.subheader("Factures avec correspondances identifiées")

            # Créer des boutons cliquables pour chaque facture
            matches_df = pd.DataFrame(matches)

            # Création d'une copie pour l'exportation Excel
            export_df = matches_df.copy()

            # Afficher le tableau avec les liens
            for i, row in matches_df.iterrows():
                invoice_name = row['Facture']
                col1, col2, col3, col4 = st.columns([1, 1, 2, 1])

                with col1:
                    # Le nom de la facture est un bouton cliquable
                    if st.button(f"📄 {invoice_name}", key=f"invoice_{i}", help="Cliquez pour voir la facture"):
                        view_invoice(invoice_name)

                with col2:
                    st.write(row['Relevé bancaire'])

                with col3:
                    st.write(row['Détail transaction'])

                with col4:
                    st.write(row['Ligne transaction'])

                st.markdown("---")

            # Bouton de téléchargement pour les correspondances
            st.markdown(
                get_download_link(export_df, 'correspondances_factures.xlsx', 'Télécharger les correspondances (Excel)'),
                unsafe_allow_html=True
            )
        else:
            st.info("Aucune correspondance trouvée.")

        # Factures sans correspondance
        if unmatched:
            st.subheader("Factures sans correspondance")
            unmatched_df = pd.DataFrame(unmatched)

            # Afficher les factures sans correspondance avec possibilité de les visualiser
            for i, row in unmatched_df.iterrows():
                invoice_name = row['Facture']
                col1, col2 = st.columns([1, 3])

                with col1:
                    if st.button(f"📄 {invoice_name}", key=f"unmatched_{i}", help="Cliquez pour voir la facture"):
                        view_invoice(invoice_name)

                st.markdown("---")

            # Bouton de téléchargement pour les factures sans correspondance
            st.markdown(
                get_download_link(unmatched_df, 'factures_sans_correspondance.xlsx', 'Télécharger les factures sans correspondance (Excel)'),
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

# Pied de page
st.markdown("---")
st.markdown("© 2025 - Application de correspondance automatique factures/relevés bancaires developper par les etudiants de l'Hetic")
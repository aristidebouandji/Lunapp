# Correspondance Automatique Factures / Relevés Bancaires

Cette application Streamlit permet d'automatiser la correspondance entre les factures (au format image) et les relevés bancaires (au format CSV). Elle utilise l'IA pour extraire les informations des factures et les faire correspondre avec les transactions bancaires.

Aperçu de l'application ---> https://lunapp-app-hetic.streamlit.app/

## Fonctionnalités

- Extraction automatique des données des factures (date, montant, fournisseur, numéro de facture)
- Analyse des relevés bancaires au format CSV
- Algorithme de correspondance en cascade basé sur plusieurs critères (montant, date, similarité du fournisseur)
- Génération de rapports avec statistiques de correspondance
- Export des résultats en format Excel
- Visualisation des factures directement dans l'interface

## Prérequis

- Python 3.8 ou supérieur
- Clé API Mistral (pour l'extraction de données des factures)

## Installation

1. Clonez ce dépôt :
```bash
git clone https://github.com/votre-username/correspondance-factures-releves.git
cd correspondance-factures-releves
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

3. Créez un fichier `.env` à la racine du projet avec votre clé API Mistral :
```
MISTRAL_API_KEY=votre_cle_api_mistral
```

## Utilisation

### Lancement de l'application

```bash
streamlit run interface1.py
```

### Guide d'utilisation

1. **Importation des fichiers**
   - Dans la zone "Importation des relevés bancaires", importez vos fichiers CSV
   - Dans la zone "Importation des factures", importez vos images de factures (JPG, PNG)

2. **Lancement de l'analyse**
   - Cliquez sur le bouton "ANALYSER ET TROUVER LES CORRESPONDANCES"
   - L'application va :
     - Extraire les données des factures (utilisation de l'API Mistral)
     - Analyser les relevés bancaires
     - Trouver les correspondances selon les règles définies

3. **Consultation des résultats**
   - Visualisez les statistiques (nombre de factures traitées, correspondances trouvées, etc.)
   - Examinez les correspondances trouvées
   - Consultez les factures sans correspondance
   - Cliquez sur les noms des factures pour les visualiser

4. **Export des résultats**
   - Utilisez les boutons de téléchargement pour exporter les résultats en Excel

## Structure des fichiers d'entrée

### Format des relevés bancaires (CSV)

L'application tentera de détecter automatiquement les colonnes suivantes :
- **Date** : date de la transaction (formats supportés : YYYY-MM-DD, DD/MM/YYYY, etc.)
- **Montant** : montant de la transaction
- **Description/Vendor** : nom du fournisseur ou description de la transaction
- **Devise** (optionnel) : devise de la transaction (par défaut : EUR)

### Format des factures

L'application accepte les formats suivants :
- JPG / JPEG
- PNG

Les factures doivent être lisibles pour que l'extraction automatique fonctionne correctement.

## Paramètres de correspondance

Les règles de correspondance sont définies dans la variable `MATCHING_RULES` :

```python
MATCHING_RULES = {
    'montant_tolerance': 0.01,    # Tolérance sur le montant (en valeur absolue)
    'jours_tolerance': 3,         # Tolérance sur la date (en jours)
    'seuil_similarite': 0.7       # Seuil minimal de similarité pour les noms des vendeurs.
}
```

## Pour les développeurs

### Structure du code

Le fichier principal `interface1.py` contient l'ensemble du code de l'application. Voici les principales fonctions :

- `process_invoices(image_folder)` : Traite les images de factures avec l'API Mistral
- `process_transactions(csv_folder)` : Traite les relevés bancaires CSV
- `find_matches(invoice, transactions)` : Algorithme de correspondance en cascade
- `calculate_vendor_similarity(vendor1, vendor2)` : Calcule la similarité entre deux noms de fournisseurs
- `generate_report(results)` : Génère le rapport de correspondance

### Gestion du rate limiting

L'application intègre un mécanisme de backoff exponentiel pour gérer les limites de l'API Mistral :
- Délai initial entre les requêtes
- Augmentation du délai en cas d'erreur 429 (rate limit exceeded)
- Réessais en cas d'échec

### Modification du modèle d'extraction

Si vous souhaitez utiliser un autre modèle pour l'extraction des données des factures, modifiez la variable `model` dans la fonction `process_invoices()` :

```python
model = "votre-modele-pixtral"
```

### Personnalisation de l'interface

L'interface utilise Streamlit et peut être personnalisée en modifiant le CSS dans la section `st.markdown("""<style>...""")`.

## Limitations connues

- Le traitement des factures est limité par les restrictions d'API de Mistral
- Certains formats de date peuvent ne pas être correctement reconnus
- La qualité de l'extraction dépend de la lisibilité des factures

## Dépendances principales

- `streamlit` : Interface utilisateur
- `pandas` : Traitement des données tabulaires
- `mistralai` : Client API pour l'extraction des données de factures
- `scikit-learn` : Calcul de similarité entre textes
- `Pillow` : Traitement d'images


## Contributeurs

- BOUANDJI Josué Aristide
- KONATE Almamy
- NOUHO Sylla
- BAH Idiatou
- AGBOGBEHOUN Dhelali
- BENMOKHTAR Khadidja

---

Développé par des étudiants de l'HETIC © 2025

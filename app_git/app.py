import numpy as np
from PIL import Image
import requests
import json
import matplotlib.pyplot as plt
import shap
import pickle
import pandas as pd
import time
from io import BytesIO
import streamlit as st
from streamlit.components.v1 import html

# Code pour afficher le logo dans la sidebar
st.sidebar.image('app_git/openclassroom.png', use_column_width=True)

# URL "Raw" de votre classificateur LGBM sur GitHub
url_classifier = 'https://raw.githubusercontent.com/BelWalL/Modele_scoring_OC/main/api_git/lgbm_classifier.pkl'

# Télécharger le contenu binaire du classificateur LGBM depuis GitHub
response = requests.get(url_classifier)
if response.status_code == 200:
    # Créer un objet similaire à un fichier en mémoire à partir du contenu binaire téléchargé
    classifier_content = BytesIO(response.content)

    # Désérialiser le classificateur LGBM à partir de cet objet
    classifier = pickle.load(classifier_content)
    print("Classificateur chargé avec succès.")
else:
    print(f"Erreur lors du téléchargement du classificateur: {response.status_code}")


# import robust scaler fit on data_train

# URL "Raw" de votre RobustScaler sur GitHub
url_scaler = 'https://raw.githubusercontent.com/BelWalL/Modele_scoring_OC/main/api_git/lgbm_robust_scaler.pkl'

# Télécharger le scaler depuis GitHub
response_scaler = requests.get(url_scaler)
scaler_content = BytesIO(response_scaler.content)
# Charger le RobustScaler
robust_scaler_tuple = pickle.load(scaler_content)
robust_scaler = robust_scaler_tuple[1]

# import df test
file_url = 'https://raw.githubusercontent.com/BelWalL/Modele_scoring_OC/main/api_git/df_prod_imp.csv'
app_test = pd.read_csv(file_url)
#print(app_test["DAYS_BIRTH"].describe())

# api url
api_url = "https://apiscoringoc-71e1b1ce78fe.herokuapp.com"


def get_customers_ids():
    # list of customers ids
    customers_ids = requests.get(api_url + "/api/v1/customers")
    content = json.loads(customers_ids.content.decode('utf-8'))
    return content


def get_customer_values(customer_id):
    try:
        # Envoyer une requête GET à l'API pour obtenir les valeurs des paramètres pour un client spécifique
        response = requests.get(f"{api_url}/api/v1/customers/{customer_id}")

        # Vérifier le statut de la réponse
        if response.status_code == 200:
            # Si la requête a réussi, convertir la réponse de JSON en objet Python et la retourner
            content = response.json()
            return content
        elif response.status_code == 404:
            # Si l'ID du client n'est pas trouvé, retourner un message indiquant que le client n'existe pas
            return {"error": "Le client spécifié n'existe pas dans notre base de données."}
        else:
            # Pour tous les autres codes d'erreur HTTP, retourner un message d'erreur général
            return {"error": "Une erreur est survenue lors de la récupération des informations du client."}
    except requests.RequestException as e:
        # Gérer les exceptions levées par les requêtes (par ex., problèmes de réseau)
        return {"error": f"Une erreur est survenue lors de la communication avec l'API : {str(e)}"}


def get_features_selected():
    # Obtenir la liste de toutes les colonnes du dataframe
    features_selected_list = app_test.columns.tolist()

    # Supprimer 'SK_ID_CURR' de la liste si présent
    if 'SK_ID_CURR' in features_selected_list:
        features_selected_list.remove('SK_ID_CURR')

    # Retourner la liste des features après suppression
    return features_selected_list

# Obtention de la liste des features sélectionnées
features_selected = get_features_selected()

def get_customer_shap_values(data_df):
    # Assurez-vous que data_df contient uniquement les features nécessaires pour la prédiction
    features_selected = data_df.columns.tolist() # feature_names

    # Extraire le scaler de la pipeline pour mise à l'échelle manuelle des données
    scaler = classifier.named_steps['scaler']
    lgbm_model = classifier.named_steps['classifier']

    # Appliquer la mise à l'échelle sur les données
    scaled_data = scaler.transform(data_df) # X_test_transformed
    # Initialiser l'Explainer SHAP pour le modèle LGBM extrait
    explainer = shap.Explainer(lgbm_model, scaled_data)

    scaled_data_df = pd.DataFrame(scaled_data, columns=features_selected)
    shap_values_ = explainer(scaled_data_df, check_additivity=False)

    fig, ax = plt.subplots()
    shap.plots.bar(shap_values_)
    st.pyplot(fig)



    #return shap_values_


def get_predicted_score(customer_id):
    # Construction de l'URL avec le customer_id dynamique
    url = f"{api_url}/api/v1/customers/{customer_id}/pred_score"

    # Envoyer la requête GET à l'API
    response = requests.get(url)

    # Vérifier si la requête a réussi
    if response.status_code == 200:
        # Convertir la réponse de JSON en objet Python et la retourner
        content = response.json()
        return content
    else:
        # En cas d'erreur, retourner un message d'erreur
        return {"error": f"Une erreur est survenue lors de la récupération du score prédit. Code d'erreur : {response.status_code}"}


def request_prediction(api_url_calc, data, max_retries=3):
    headers = {"Content-Type": "application/json"}
    data_dict = data.to_dict(orient='records')[0]
    data_json = {'data': data_dict}
    for attempt in range(max_retries):
        try:
            response = requests.request(
                method='POST', headers=headers, url=api_url_calc, json=data_json, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ReadTimeout:
            print(f"Request timed out. Retrying... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(2)  # Add a small delay before retrying
    raise Exception("Maximum retries reached. Unable to get a response.")

    return response.json()


def construire_jauge_score(score_remboursement_client):
    # Define the gauge ranges and colors
    # Adjust the gauge range to include your new threshold at 0.4
    gauge_ranges = [0, 0.4, 1]
    gauge_colors = ["#3C8B4E","#E0162B"]  # Vert pour "bien" (score <= 0.4), Rouge pour "mauvais" (score > 0.4)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 1))

    # Plot the gauge ranges with colors
    for i in range(1, len(gauge_ranges)):
        ax.fill_betweenx([0, 1], gauge_ranges[i - 1], gauge_ranges[i], color=gauge_colors[i - 1])

    # Plot the current value on the gauge
    ax.plot([score_remboursement_client, score_remboursement_client], [0, 1], color="black", linewidth=2)
    # Adjust this line to plot the threshold at 0.4 instead of 0.55
    ax.plot([0.4, 0.4], [0, 1], color="#A31D2B", linewidth=2, linestyle='--')

    # Set axis limits and labels
    ax.set_xlim(0, 1)
    ax.set_title("Probabilité de remboursement du crédit")
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels([f'{tick:.1f}' for tick in np.arange(0, 1.1, 0.1)])
    ax.set_yticks([])

    return fig

# Récupération des identifiants des clients via l'API
customers_ids = get_customers_ids()

# Titre de la sidebar pour la sélection du client
st.sidebar.markdown('<p style="font-family: San Francisco, sans-serif; font-size: 16px; color: darkblue; font-weight: bold;">'
                    'Sélectionner le client à tester :</p>', unsafe_allow_html=True)

# Sélection d'un client à partir de la sidebar
customer_id = st.sidebar.selectbox('', customers_ids)

# Continuation après la sélection du client dans la sidebar
if customer_id:
    # Récupérer les données détaillées du client
    customer_data = get_customer_values(customer_id)

    if 'error' not in customer_data:
        # Préparation des données pour l'affichage
        age_years = round(abs(customer_data['DAYS_BIRTH']) / 365,0)
        gender = "Masculin" if customer_data.get('CODE_GENDER_M', 0) == 1 else "Féminin"

        # Rond des scores à deux décimales
        ext_source_2_rounded = round(customer_data.get('EXT_SOURCE_2', 0), 3)
        ext_source_3_rounded = round(customer_data.get('EXT_SOURCE_3', 0), 3)

        # Création d'un DataFrame pour le tableau d'affichage sans l'index
        data_for_display = {
            "Nom de Variable": ["Âge du Client (années)", "Score Externe 2", "Score Externe 3", "Genre"],
            "Valeur": [age_years,
                       ext_source_2_rounded,
                       ext_source_3_rounded,
                       gender]
        }

        df_display = pd.DataFrame(data_for_display)
        df_display["Valeur"] = df_display["Valeur"].astype(str)


        # Affichage du tableau des variables sélectionnées dans la sidebar sans l'index
        st.sidebar.write("## Détails du Client Sélectionné")
        st.sidebar.table(df_display.set_index("Nom de Variable"))

    else:
        # Afficher l'erreur dans la sidebar si le client n'est pas trouvé ou autre erreur API
        st.sidebar.error(customer_data.get('error'))
else:
    # Message d'encouragement à sélectionner un client dans la sidebar si ce n'est pas encore fait
    st.sidebar.warning("Veuillez sélectionner un client pour afficher les détails.")
# Titre principal de la page
st.title("Dashboard d'évaluation de crédit")

# Bouton et prédiction de risque de crédit en utilisant l'endpoint GET
if customer_id:
    risk_url = f"{api_url}/api/v1/customers/{customer_id}/pred_score"
    response = requests.get(risk_url)

    if response.status_code == 200:
        risk_content = response.json()
        proba_risk_str = risk_content["Proba_risk"]
        proba_risk_value = float(proba_risk_str.split(":")[1].strip().replace("%",""))

        if proba_risk_value > 40:
            st.markdown("<p style='font-family: San Francisco, sans-serif; font-size:24px; color:red;'>Crédit refusé</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='font-family: San Francisco, sans-serif; font-size:24px; color:green;'>Crédit accordé</p>", unsafe_allow_html=True)

        st.write(f'Le risque de défaut pour ce client est de {proba_risk_value:.2f}%.')
        st.write('Le seuil de décision est de 40%.')
        jauge_score = construire_jauge_score(proba_risk_value / 100) # Ajustez cette fonction si nécessaire
        st.pyplot(jauge_score)
    else:
        st.error("Impossible de récupérer la probabilité de risque de crédit pour ce client.")




def plot_feature_distribution(df, feature, customer_feature_value=None):
    plt.figure(figsize=(10, 6))
    plt.hist(df[feature], bins=30, alpha=0.7, label="Distribution globale", color="blue")

    if customer_feature_value is not None:
        plt.axvline(customer_feature_value, color='red', linestyle='--', label=f"Valeur du client ({feature})")

    plt.title(f"Distribution de {feature}")
    plt.xlabel(feature)
    plt.ylabel("Nombre d'observations")
    plt.legend()
    plt.grid(True)


# Sélection d'une caractéristique pour voir sa distribution
selected_feature = st.selectbox(
    "Sélectionnez une caractéristique pour voir la distribution:",
    features_selected
)

if customer_id and selected_feature:
    # Récupérer les données du client sélectionné
    customer_data = get_customer_values(customer_id)

    if 'error' not in customer_data:
        # Récupérer la valeur de la caractéristique pour le client sélectionné
        customer_feature_value = customer_data.get(selected_feature, None)

        # Afficher la distribution de la caractéristique sélectionnée et la valeur pour ce client
        plot_feature_distribution(app_test, selected_feature, customer_feature_value)

        # Utiliser st.pyplot() pour afficher le graphe dans Streamlit
        st.pyplot(plt)

# Charger l'image de feature importance globale
feature_importance = Image.open('app_git/feature_importance_globale.png')

# Créer une case à cocher
show_image = st.checkbox("Afficher l'importance globale des variables")

# Afficher l'image uniquement si la case à cocher est cochée
if show_image:
    st.image(feature_importance, caption="Importance globale des variables", use_column_width=True)

# Créer une case à cocher pour décider d'afficher ou non le rapport DataDrift
show_datadrift_report = st.checkbox("Afficher le rapport DataDrift")

if show_datadrift_report:
    # Lire le contenu du fichier HTML
    with open('app_git/rapport_data_drift.html', 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Créer un bouton de téléchargement pour le contenu HTML
    st.download_button(
        label="Télécharger le rapport DataDrift",
        data=html_content,
        file_name="rapport_data_drift.html",
        mime="text/html"
    )














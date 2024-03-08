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


# URL "Raw" de votre classificateur LGBM sur GitHub
url_classifier = 'https://raw.githubusercontent.com/BelWalL/Projet7OC/main/api_git/lgbm_classifier.pkl'

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
url_scaler = 'https://raw.githubusercontent.com/BelWalL/Projet7OC/main/api_git/lgbm_robust_scaler.pkl'

# Télécharger le scaler depuis GitHub
response_scaler = requests.get(url_scaler)
scaler_content = BytesIO(response_scaler.content)
# Charger le RobustScaler
robust_scaler = pickle.load(scaler_content)

# import df test
file_url = 'https://raw.githubusercontent.com/BelWalL/Projet7OC/main/api_git/df_prod_imp.csv'
app_test = pd.read_csv(file_url)
#print(app_test.head())

# api url
api_url = "https://fastapiprojet7-4bc1fb2705e6.herokuapp.com"


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
    features_selected = data_df.columns.tolist()

    # Extraire le scaler de la pipeline pour mise à l'échelle manuelle des données
    scaler = classifier.named_steps['scaler']

    # Appliquer la mise à l'échelle sur les données
    scaled_data = scaler.transform(data_df)

    # Créer l'explainer SHAP avec le classificateur extrait de la pipeline
    explainer = shap.TreeExplainer(classifier.named_steps['classifier'])

    # Calculer les valeurs SHAP pour les données mises à l'échelle
    shap_values_list = explainer.shap_values(scaled_data)

    return shap_values_list, scaled_data, features_selected


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
        age_years = customer_data['DAYS_BIRTH']
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

# import de targuet
file_url_targuet = 'https://raw.githubusercontent.com/BelWalL/Projet7OC/main/api_git/df_prod_imp_with_predictions.csv'
predictions_df  = pd.read_csv(file_url_targuet)

# Fusionner les prédictions avec le dataset principal sur SK_ID_CURR
app_test_with_predictions = pd.merge(app_test, predictions_df, on='SK_ID_CURR', how='left')


def build_histogram(df, feature_selected_1, predicted_target):
    fig, ax = plt.subplots()
    # Filtrer les données basées sur la prédiction cible
    df_target = df[df['Predicted_Class'] == predicted_target]
    # Supposons que vous ayez une manière de récupérer la valeur actuelle de la feature pour le client actuel
    customer_app_value = customer_data.get(feature_selected_1)

    # Vérification pour éviter les valeurs <= 0 avec l'échelle log
    df_target = df_target[df_target[feature_selected_1] > 0]

    ax.hist(df_target[feature_selected_1], bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)

    # Appliquer une échelle logarithmique pour l'axe X
    ax.set_xscale('log')

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xlabel(feature_selected_1)
    ax.set_ylabel('Fréquence')
    title = 'Crédit accordé' if predicted_target == 0 else 'Crédit refusé'
    ax.set_title(f'Distribution de la variable {feature_selected_1} pour les clients ayant un {title}')

    if customer_app_value is not None and customer_app_value > 0:  # Assurez-vous également ici que la valeur n'est pas <= 0
        ax.axvline(x=customer_app_value, color='red', linestyle='--', linewidth=2, label='Position du client')
    ax.legend()

    return fig


feature_selected_1 = st.selectbox(
    'Sélectionner une variable pour visualiser sa distribution selon le score du crédit',
    features_selected
)

predicted_target = st.radio("Sélectionnez le type de clients pour la visualisation", (0, 1))

histogram_fig = build_histogram(app_test_with_predictions, feature_selected_1, predicted_target)
st.pyplot(histogram_fig)




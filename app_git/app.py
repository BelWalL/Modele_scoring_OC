import streamlit as st
import requests

# Définissez l'URL de votre API ici
api_url = "https://fastapiprojet7-4bc1fb2705e6.herokuapp.com"

# Cette fonction utilise `api_url` et attend seulement `customer_id` comme paramètre
def get_customer_prediction(customer_id):
    response = requests.get(f"{api_url}/api/v1/customers/{customer_id}/pred_score")
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Erreur lors de la récupération des données. Code: {response.status_code}"}

#Titre de tableau de bord
st.title("Dashboard de Prédiction de Crédit")

# Créez un champ de texte pour l'entrée de l'ID client
customer_id = st.text_input("Entrez l'ID du client pour prédire le risque de crédit:", '')

# Lorsque l'utilisateur appuie sur le bouton "Prédire", appelez la fonction avec l'ID fourni
if st.button("Prédire"):
    if customer_id:
        prediction = get_customer_prediction(customer_id)  # Appelez la fonction avec le bon paramètre
        if "error" not in prediction:
            st.success(f"Résultat de la prédiction : {prediction}")
        else:
            st.error("Une erreur est survenue.")
    else:
        st.error("Veuillez entrer un ID de client valide.")
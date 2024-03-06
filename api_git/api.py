from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import uvicorn

# Initialisation de l'application FastAPI
app = FastAPI()

# Charger le modèle de prédiction depuis le fichier sérialisé
with open("lgbm_classifier.pkl", "rb") as file:
    classifier = pickle.load(file)

# Modèle Pydantic pour valider les données reçues par l'API pour les détails des clients
class CustomerData(BaseModel):
    SK_ID_CURR: int
    PAYMENT_RATE: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    DAYS_BIRTH: float
    INSTAL_AMT_PAYMENT_MIN: float
    AMT_ANNUITY: float
    ANNUITY_INCOME_PERC: float
    DAYS_EMPLOYED: float
    DAYS_EMPLOYED_PERC: float
    PREV_CNT_PAYMENT_MEAN: float
    CODE_GENDER_M: float
    APPROVED_CNT_PAYMENT_MEAN: float
    INSTAL_DAYS_ENTRY_PAYMENT_MEAN: float
    INSTAL_DPD_MEAN: float
    AMT_CREDIT: float
    AMT_GOODS_PRICE: float
    DAYS_ID_PUBLISH: float
    POS_MONTHS_BALANCE_SIZE: float
    ACTIVE_DAYS_CREDIT_MAX: float
    ACTIVE_DAYS_CREDIT_ENDDATE_MAX: float

# Modèle Pydantic pour valider les données reçues par l'API pour la prédiction
class PredictionRequest(BaseModel):
    PAYMENT_RATE: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    DAYS_BIRTH: float
    INSTAL_AMT_PAYMENT_MIN: float
    AMT_ANNUITY: float
    ANNUITY_INCOME_PERC: float
    DAYS_EMPLOYED: float
    DAYS_EMPLOYED_PERC: float
    PREV_CNT_PAYMENT_MEAN: float
    CODE_GENDER_M: float
    APPROVED_CNT_PAYMENT_MEAN: float
    INSTAL_DAYS_ENTRY_PAYMENT_MEAN: float
    INSTAL_DPD_MEAN: float
    AMT_CREDIT: float
    AMT_GOODS_PRICE: float
    DAYS_ID_PUBLISH: float
    POS_MONTHS_BALANCE_SIZE: float
    ACTIVE_DAYS_CREDIT_MAX: float
    ACTIVE_DAYS_CREDIT_ENDDATE_MAX: float

# Charger le dataframe de test
app_test = pd.read_csv('df_prod_imp.csv')



# Route racine pour afficher un message de bienvenue et des instructions
@app.get("/")
def home():
    return {
        "message": "Bienvenue sur l'API !",
        "instructions": {
            "customers_ids": "Pour retrouver les identifiants clients, utilisez /api/v1/customers",
            "customer_features": "Pour retrouver les caractéristiques d'un client, utilisez /api/v1/customers/{customer_id}",
            "predict_risk": "Pour afficher la probabilité du risque de crédit pour un client, utilisez /api/v1/customers/{customer_id}/pred_score",
            "predict_post": "Pour prédire en utilisant des données spécifiques via une requête POST, utilisez /api/v1/predict avec les données requises au format JSON."
        }
    }


# Route pour récupérer la liste des identifiants clients
@app.get("/api/v1/customers")
def customers_ids():
    customers = app_test['SK_ID_CURR'].tolist()
    return customers


# Route pour récupérer les caractéristiques d'un client
@app.get("/api/v1/customers/{customer_id}")
def columns_values(customer_id: int):
    if not app_test['SK_ID_CURR'].isin([customer_id]).any():
        raise HTTPException(status_code=404, detail="Customer ID not found")
    content = app_test[app_test['SK_ID_CURR'] == customer_id].iloc[0].to_dict()
    return content


# Route pour prédire le risque de crédit d'un client spécifique
@app.get("/api/v1/customers/{customer_id}/pred_score")
def predict_customer(customer_id: int):
    if not app_test['SK_ID_CURR'].isin([customer_id]).any():
        raise HTTPException(status_code=404, detail="Customer ID not found")
    proba = classifier.predict_proba(app_test.loc[app_test['SK_ID_CURR'] == customer_id, app_test.columns[:-1]])
    proba_risk = (proba[:,1]*100).item()
    return {"Proba_risk": f"Probabilité que le crédit soit refusé : {proba_risk:.2f}%"}


# Fonction pour effectuer des prédictions
def predict_function(input_data_df: pd.DataFrame):
    # Utilisation du modèle pour prédire
    proba = classifier.predict_proba(input_data_df)
    return proba[:, 1]  # Retourner la probabilité de la classe positive


# Route pour prédire en utilisant une requête POST avec des données spécifiques
@app.post("/api/v1/predict")
def predict(predict_request: PredictionRequest):
    # Conversion directe de l'objet Pydantic en dictionnaire pour la construction du DataFrame
    input_data_df = pd.DataFrame([predict_request.dict()])
    proba = classifier.predict_proba(input_data_df).tolist()
    result = {
        "prediction": proba,
        "message": "Prediction successful"
    }
    return result



# Démarrer le serveur Uvicorn si le script est exécuté directement
if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000)


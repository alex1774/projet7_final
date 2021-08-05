
import pandas as pd
# import pickle
import numpy as np

import plotly.express as px
import dash
#from Dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
# from secret import access_key,secret_access_key
# import awswrangler as wr
# import boto3
from preprocess_data_p7 import *
import joblib
# from io import BytesIO

# s3 = boto3.resource('s3',
#          aws_access_key_id=access_key,
#          aws_secret_access_key= secret_access_key)
#
# data = wr.s3.read_csv(path = 's3://amoreauopenclassrooms/application_train.csv')
data = pd.read_csv('data_dash.csv')

data.set_index('SK_ID_CURR',inplace=True)


data = transform_data_to_predict(data)

#
# s3 = boto3.resource('s3')
# bucket_str = "amoreauopenclassrooms"
# bucket_key = "logistic_reg_tuned.pkl"
# with BytesIO() as data_transfert:
#     s3.Bucket(bucket_str).download_fileobj(bucket_key, data_transfert)
#     data_transfert.seek(0)    # move back to the beginning after writing
#     model_log_reg = joblib.load(data_transfert)
model_log_reg = joblib.load('model_final.pkl')

def create_card(title, content,color):
    card = dbc.Card(
        dbc.CardBody(
            [
                html.H4(title, className="card-title"),
                html.Br(),
                html.H2(id=content, className="card-subtitle"),
                html.Br(),

                ]
        ),
        color=color, inverse=True
    )
    return(card)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

gender_card = create_card('Genre du client', 'gender', 'info')
car_card = create_card('Possède une voiture ?', 'car', 'info')
realty_card = create_card('Possède une maison ?', 'realty', 'info')
childs_card = create_card('Nombre d enfants', 'childs', 'info')
fam_status_card = create_card('Situation familiale', 'fam_status', 'info')
age_card = create_card('Age du client', 'age', 'info')
contrat_card = create_card('Type de prêt :', 'contrat', 'info')
revenu_card = create_card('Revenu total :', 'revenu', 'info')
montant_credit_card = create_card('Montant emprunté :', 'montant_credit', 'info')
annuite_card = create_card('Montant annuité :', 'annuite', 'info')
prix_achat_card = create_card('Prix de l achat :', 'prix_achat', 'info')
proba_1_card = create_card('calcul du défaut de paiement (%) :', 'proba_1', 'info')
average_note_card = create_card('Note du client issu d organismes extérieurs (../1):', 'average_note', 'info')
age_car_card = create_card('Age de la voiture du client (années) :', 'age_car', 'info')
age_job_card = create_card('Ancienneté du client sur son poste (années) :', 'age_job', 'info')
recalcul_proba_1_card = create_card('Recalcul de la probabilité de défaut de paiement (%):', 'recalcul_proba_1', 'info')
classe_client_card = create_card('client en défaut de paiement :', 'classe_client', 'info')


liste_options = {'note moyenne ext.': 'AVG_NOTE',
                 'age véhicule': 'OWN_CAR_AGE',
                 'age client': 'DAYS_BIRTH',
                 'ancienneté sur le poste': 'DAYS_EMPLOYED',
                 'prix achat bien': 'AMT_GOODS_PRICE',
                 'montant crédit': 'AMT_CREDIT',
                 'revenu total': 'AMT_INCOME_TOTAL'}

app.layout = html.Div([
    html.I("Merci de rentrer l'ID client, svp."),
    html.Br(),
    dcc.Input(id='id_input', type='number', value=149005, style={'marginRight': '10px'}),
    html.Hr(),
    html.H3(" Informations générales :"),
    html.P(),
    dbc.Row([dbc.Col(id='gender_card', children=[gender_card], lg=4, width=4),
             dbc.Col(id='age_card', children=[age_card], lg=4, width=4),
             dbc.Col(id='fam_status_card', children=[fam_status_card], lg=4, width=4)]),
    html.P(),
    dbc.Row([dbc.Col(id='childs_card', children=[childs_card], lg=4, width=4),
             dbc.Col(id='car_card', children=[car_card], lg=4, width=4),
             dbc.Col(id='realty_card', children=[realty_card], lg=4, width=4)]),

    html.P(),
    dbc.Row([dbc.Col(id='age_car_card', children=[age_car_card], lg=4, width=4),
             dbc.Col(id='age_job_card', children=[age_job_card], lg=4, width=4)]),

    html.Hr(),
    html.H3("Informations financières"),
    html.P(),
    dbc.Row([dbc.Col(id='contrat_card', children=[contrat_card], lg=4, width=4),
             dbc.Col(id='revenu_card', children=[revenu_card], lg=4, width=4),
             dbc.Col(id='montant_credit_card', children=[montant_credit_card], lg=4, width=4)]),
    html.P(),
    dbc.Row([dbc.Col(id='annuite_card', children=[annuite_card], lg=4, width=4),
             dbc.Col(id='prix_achat_card', children=[prix_achat_card], lg=4, width=4),
             dbc.Col(id='average_note_card', children=[average_note_card], lg=4, width=4)]),
    html.Hr(),
    html.H3("Probabilité de défaut de paiement"),
    html.P(),
    dbc.Row([#dbc.Col(id='proba_1_card', children=[proba_1_card], lg=4, width=6),
             #dbc.Col(id='recalcul_proba_1_card', children=[recalcul_proba_1_card], lg=4, width=12),
             dbc.Col(id='classe_client_card', children=[classe_client_card], lg=4, width=6),
             ]),
    html.P(),
    html.P(),
    html.H4(
        "Veuillez changer les variables ci-dessous pour recalculer la probabilité pour le client d'etre en défaut de paiement"),

    html.P(),
    html.P(),

    html.Div(id='slider-output-ext_source'),
    html.Br(),
    dcc.Slider(
        id='slider_ext_source',
        min=0,
        max=1,
        step=0.01,
        value=0.5),

    html.Div(id='slider-output-age_car'),
    html.Br(),
    dcc.Slider(
        id='slider_age_car',
        min=0,
        max=100,
        step=1,
        value=4),

    html.Div(id='slider-output-age_client'),
    html.Br(),
    dcc.Slider(
        id='slider_age_client',
        min=20,
        max=70,
        step=1,
        value=44),

    html.Div(id='slider-output-anciennete'),
    html.Br(),
    dcc.Slider(
        id='slider_anciennete',
        min=0,
        max=50,
        step=1,
        value=5),

    html.Div(id='slider-output-prix_bien'),
    html.Br(),
    dcc.Slider(
        id='slider_prix_bien',
        min=50000,
        max=4000000,
        step=25000,
        value=540000),

    html.Div(id='slider-output-montant_pret'),
    html.Br(),
    dcc.Slider(
        id='slider_montant_pret',
        min=50000,
        max=4000000,
        step=25000,
        value=540000),

    html.Div(id='slider-output-revenu'),
    html.Br(),
    dcc.Slider(
        id='slider_revenu',
        min=25000,
        max=4500000,
        step=25000,
        value=175000),
    html.P(),

    dbc.Col(id='recalcul_proba_1_card', children=[recalcul_proba_1_card], lg=4, width=12),

    html.Hr(),
    html.H3("Graphique interactif : "),
    html.H5(
        "L'objectif est de pouvoir visualiser pour un ensemble de variables, les clients en défaut de paiement par rapport aux autres"),
    html.Div([
        html.Label(['Graphique interactif']),
        dcc.Dropdown(id='dropdown_1',
                     options=[{"label": x, "value": x} for x in liste_options.keys()],
                     value='note moyenne ext.',
                     clearable=False),
        dcc.Graph(id="bar-chart"),
    ], style={'height': 600}),
])


@app.callback(
    Output("gender", "children"),
    Output("car", "children"),
    Output("realty", "children"),
    Output("childs", "children"),
    Output("fam_status", "children"),
    Output("age", "children"),
    Output("contrat", "children"),
    Output("revenu", "children"),
    Output("montant_credit", "children"),
    Output("annuite", "children"),
    Output("prix_achat", "children"),
    #Output("proba_1", "children"),
    Output("average_note", "children"),
    Output("age_car", "children"),
    Output("age_job", "children"),
    Output("classe_client","children"),
    Input("id_input", "value"),
)
def update_output_1(id_input):
    def infos_vitrine(id_client=149005):
        (gender, car, realty, childs, fam_status, age, contrat, revenu, montant_credit,
         annuite, prix_achat,age_job) = data.loc[
            id_client, ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
                     "CNT_CHILDREN", "NAME_FAMILY_STATUS", "DAYS_BIRTH", "NAME_CONTRACT_TYPE", "AMT_INCOME_TOTAL",
                     "AMT_CREDIT",
                     "AMT_ANNUITY", "AMT_GOODS_PRICE","DAYS_EMPLOYED"]]
        # correction 'gender' :
        if gender == 'F':
            gender = 'Femme'
        if gender == 'M':
            gender = 'Homme'
        if gender == 'XNA':
            gender = 'Autre'

        # correction 'car':
        if car == 'Y':
            car = 'O'

        # correction 'realty':
        if realty == 'Y':
            realty = 'O'

        # correction 'fam_status':
        if fam_status == 'Married':
            fam_status = 'Marrié.e'
        if fam_status == 'Single / not married':
            fam_status = 'Célibataire / non marié.e'
        if fam_status == 'Civil marriage':
            fam_status = 'Union civile'
        if fam_status == 'Separated':
            fam_status = 'Séparé.e'
        if fam_status == 'Widow':
            fam_status = 'Veuf.ve'

        # correction contrat:
        if contrat == 'Cash loans':
            contrat = 'Prêt'
        if contrat == 'Revolving loans':
            contrat = 'Crédit renouvelable'

        # correction age :
        age = round(age/-365)

        # correction age_job
        age_job = round(age_job/-365)

        return (gender, car, realty, childs, fam_status, age, contrat, revenu, montant_credit,
                annuite, prix_achat,age_job)

    (gender, car, realty, childs, fam_status, age, contrat, revenu, montant_credit,
     annuite, prix_achat,age_job)=infos_vitrine(id_input)
    #data_predict_1 = data.drop(columns=['TARGET']).set_index('SK_ID_CURR')
    #infos_1 = data_predict_1.loc[data_predict_1.index == id_input]

    #proba_1 = f" {round(100 * model_log_reg.predict_proba(data.loc[data.index == id_input].drop(columns=['TARGET']))[0][1], 2)} %"

    classe = model_log_reg.predict(data.drop(columns=['TARGET']).loc[data.index == id_input])[0]
    if classe == True:
        classe_client = 'Client en défaut de paiement'
    if classe == False:
        classe_client = 'Client n etant pas en défaut de paiement'

    average_note = round(float(data.loc[data.index == id_input].AVG_NOTE), 2)
    age_car = int(data.loc[data.index == id_input].OWN_CAR_AGE)


    return gender, car, realty, childs, fam_status, age, contrat, revenu, montant_credit, annuite, prix_achat, average_note, age_car, age_job,classe_client


@app.callback(
    Output('slider_ext_source', 'value'),
    Output('slider_age_car', 'value'),
    Output('slider_age_client', 'value'),
    Output('slider_anciennete', 'value'),
    Output('slider_prix_bien', 'value'),
    Output('slider_montant_pret', 'value'),
    Output('slider_revenu', 'value'),
    Input("average_note", 'children'),
    Input("age_car", 'children'),
    Input("age", 'children'),
    Input("age_job", 'children'),
    Input("prix_achat", 'children'),
    Input("montant_credit", 'children'),
    Input("revenu", 'children'),
)
def update_slider(average_note, age_car, age, age_job, prix_achat, montant_credit, revenu):
    return average_note, age_car, age, age_job, prix_achat, montant_credit, revenu


@app.callback(
    Output('slider-output-ext_source', 'children'),
    Output('slider-output-age_car', 'children'),
    Output('slider-output-age_client', 'children'),
    Output('slider-output-anciennete', 'children'),
    Output('slider-output-prix_bien', 'children'),
    Output('slider-output-montant_pret', 'children'),
    Output('slider-output-revenu', 'children'),
    Input('slider_ext_source', 'value'),
    Input('slider_age_car', 'value'),
    Input('slider_age_client', 'value'),
    Input('slider_anciennete', 'value'),
    Input('slider_prix_bien', 'value'),
    Input('slider_montant_pret', 'value'),
    Input('slider_revenu', 'value')
)
def update_output_2(value_source, car_age, client_age, anciennete, prix_bien, montant_pret, revenu):
    note_ext = 'Notation du client provenant des organismes extérieurs (../1): "{}"'.format(value_source)
    age_car = 'Nombre d annees du véhicule du client : "{}"'.format(car_age)
    age_client = 'Age du client : "{}"'.format(client_age)
    temps_travail = 'Années d ancienneté sur le poste actuel : "{}"'.format(anciennete)
    bien_prix = 'Prix du bien pour lequel est souscrit le prêt : "{}"'.format(prix_bien)
    pret_montant = 'Montant du prêt : "{}"'.format(montant_pret)
    revenu = 'Revenu annuel total du client :"{}"'.format(revenu)

    return note_ext, age_car, age_client, temps_travail, bien_prix, pret_montant, revenu


@app.callback(
    Output('recalcul_proba_1', 'children'),
    Input('slider_ext_source', 'value'),
    Input('slider_age_car', 'value'),
    Input('slider_age_client', 'value'),
    Input('slider_anciennete', 'value'),
    Input('slider_prix_bien', 'value'),
    Input('slider_montant_pret', 'value'),
    Input('slider_revenu', 'value'),
    Input("id_input", "value")
)
def recalcul_proba(value_source, car_age, client_age, anciennete, prix_bien, montant_pret, revenu, id_client):

    infos = data.drop(columns=['TARGET'])
    infos = infos.loc[data.index == id_client]
    infos.at[id_client, 'AVG_NOTE'] = (value_source)
    infos.at[id_client, 'OWN_CAR_AGE'] = (car_age)
    infos.at[id_client, 'DAYS_BIRTH'] = (client_age)*-365
    infos.at[id_client, 'DAYS_EMPLOYED'] = (anciennete)*-365
    infos.at[id_client, 'AMT_GOODS_PRICE'] = (prix_bien)
    infos.at[id_client, 'AMT_CREDIT'] = (montant_pret)
    infos.at[id_client, 'AMT_INCOME_TOTAL'] = (revenu)

    return f" {round(100*model_log_reg.predict_proba(infos)[0][1], 2)} %"


@app.callback(
    Output('bar-chart', 'figure'),
    [Input('dropdown_1', 'value')]
)
def update_bar_chart(option):
    data_visu = data.copy()
    data_visu['DAYS_BIRTH']=data_visu['DAYS_BIRTH']/-365
    data_visu['DAYS_EMPLOYED']=data_visu['DAYS_EMPLOYED'].replace(to_replace=365243, value=0)
    data_visu['DAYS_EMPLOYED']=data_visu['DAYS_EMPLOYED']/-365
    mask = data_visu[[liste_options[option], 'TARGET']].groupby(by='TARGET').mean()
    fig = px.bar(mask, x=mask.index, y=liste_options[option],
                 labels={'TARGET': 'clients en défaut de paiement (1) ou non (0)',
                         'AVG_NOTE': 'note moyenne ext.',
                         'OWN_CAR_AGE': 'age véhicule',
                         'DAYS_BIRTH': 'age client',
                         'DAYS_EMPLOYED': 'ancienneté sur le poste',
                         'AMT_GOODS_PRICE': 'prix achat bien',
                         'AMT_CREDIT': 'montant crédit',
                         'AMT_INCOME_TOTAL': 'revenu total'})
    return fig


if __name__ == '__main__':
    app.run_server()



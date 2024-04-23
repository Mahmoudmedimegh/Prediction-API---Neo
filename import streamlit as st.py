import streamlit as st
import requests

st.title('Formulaire de prédiction du score client')

with st.form("client_data"):
    code_gender = st.selectbox('Gender', ['M', 'F'])
    flag_own_car = st.selectbox('Own a Car?', ['Y', 'N'])
    organization_type = st.text_input('Organization Type')
    days_birth = st.number_input('Days Birth', value=0)
    days_id_publish = st.number_input('Days ID Publish', value=0)
    sk_id_curr = st.number_input('SK ID CURR', value=0)
    reg_city_not_live_city = st.number_input('Reg City Not Live City', value=0)
    ext_source_1 = st.number_input('EXT Source 1', value=0.0)
    ext_source_2 = st.number_input('EXT Source 2', value=0.0)
    ext_source_3 = st.number_input('EXT Source 3', value=0.0)
    years_beginexploataion_mode = st.number_input('Years Begin Exploataion Mode', value=0.0)
    commonarea_mode = st.number_input('Common Area Mode', value=0.0)
    floorsmax_mode = st.number_input('Floors Max Mode', value=0.0)
    livingapartments_mode = st.number_input('Living Apartments Mode', value=0.0)
    years_build_medi = st.number_input('Years Build Medi', value=0.0)
    
    submitted = st.form_submit_button("Submit")
    if submitted:
        data = {
            "CODE_GENDER": code_gender,
            "FLAG_OWN_CAR": flag_own_car,
            "ORGANIZATION_TYPE": organization_type,
            # Ajoutez les autres champs ici
        }
        response = requests.post("http://127.0.0.1:8000/predict/", json=data)
        if response.status_code == 200:
            prediction = response.json()['prediction']
            st.write(f'Prediction: {prediction}')
        else:
            st.write("Erreur lors de la prédiction")

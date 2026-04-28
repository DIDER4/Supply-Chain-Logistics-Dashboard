"""
Dette er systemets "hjerne". Filen bygger en interaktiv hjemmeside,
hvor vi kan se data over pakker og forudsige, om nye pakker bliver forsinket.
"""

# Hent de nødvendige byggeklodser (biblioteker) til hjemmeside, data, grafer og forudsigelser
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Sæt grundlæggende indstillinger for vores hjemmeside (f.eks. bredden på skærmen)
st.set_page_config(page_title="Leveringsanalyse Dashboard", layout="wide")

# Denne funktion henter pakkedata-regnearket ind i hukommelsen. 
# @st.cache_data gør, at systemet husker dataen, så vi ikke skal læse filen forfra ved hvert klik.
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dk_parcel_delivery_basic_.csv")
        # Fortæl systemet hvilke kolonner der indeholder datoer og klokkeslæt
        df['order_datetime'] = pd.to_datetime(df['order_datetime'])
        df['dispatch_datetime'] = pd.to_datetime(df['dispatch_datetime'])
        return df
    except FileNotFoundError:
        return None

# Start med at indlæse vores pakke-data
df = load_data()

if df is None:
    st.error("Filen `dk_parcel_delivery_basic_.csv` blev ikke fundet. Sørg for, at scriptet køres fra samme mappe.")
    st.stop()

# === OVERSKRIFT PÅ HJEMMESIDEN ===
st.title("Leveringsanalyse Dashboard")

# === MENU I VENSTRE SIDE (FILTRE) ===
# Her laver vi et par menu-bokse, hvor man kan vælge, hvilke transportører og lagerbyer man vil se tal for.
st.sidebar.header("Filtre")

all_carriers = df['carrier'].unique() # Find alle unikke transportører i dataen
selected_carriers = st.sidebar.multiselect("Vælg Transportør", options=all_carriers, default=all_carriers)

all_warehouses = df['warehouse_city'].unique() # Find alle unikke lagerbyer
selected_warehouses = st.sidebar.multiselect("Vælg Distributionscenter", options=all_warehouses, default=all_warehouses)

# === ANVEND DE VALGTE FILTRE PÅ DATAEN ===
# Fjern de pakker, der ikke passer med valget i menuen, så graferne kun viser det valgte.
filtered_df = df[
    (df['carrier'].isin(selected_carriers)) &
    (df['warehouse_city'].isin(selected_warehouses))
]

# Hvis der slet ikke er valgt noget fra menuen, stopper vi her og beder brugeren om at vælge en mindstén.
if filtered_df.empty:
    st.warning("Vælg mindst én transportør og ét distributionscenter for at se data.")
    st.stop()

# === HOVEDTAL (NØGLETAL) ===
# Vi regner de største tal ud, som totalt antal pakker og forsinkelsesprocent.
st.markdown("### Nøgletal")
total_packages = len(filtered_df)
delayed_packages = filtered_df['is_delayed'].sum()
delay_rate = (delayed_packages / total_packages * 100) if total_packages > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Totale Pakker", f"{total_packages:,}")
col2.metric("Forsinkede Pakker", f"{delayed_packages:,}")
col3.metric("Forsinkelsesprocent", f"{delay_rate:.2f}%")
col4.metric("Gns. Afstand (km)", f"{filtered_df['distance_km'].mean():.1f}")

st.markdown("---")

# Visualiseringer Række 1
c1, c2 = st.columns(2)

with c1:
    st.subheader("Transportør vs. Forsinkelse")
    carrier_delay = filtered_df.groupby('carrier')['is_delayed'].mean().reset_index()
    carrier_delay['is_delayed'] *= 100
    fig1 = px.bar(carrier_delay, x='carrier', y='is_delayed', 
                  labels={'is_delayed': 'Forsinkelse (%)', 'carrier': 'Transportør'},
                  text_auto='.1f', color='carrier')
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.subheader("Distributionscenter vs. Forsinkelse")
    wh_delay = filtered_df.groupby('warehouse_city')['is_delayed'].mean().reset_index()
    wh_delay['is_delayed'] *= 100
    fig2 = px.bar(wh_delay, x='warehouse_city', y='is_delayed', 
                  labels={'is_delayed': 'Forsinkelse (%)', 'warehouse_city': 'Distributionscenter'},
                  text_auto='.1f', color='warehouse_city')
    st.plotly_chart(fig2, use_container_width=True)

# Visualiseringer Række 2
c3, c4, c5 = st.columns(3)

with c3:
    st.subheader("Højsæson vs. Forsinkelse")
    peak_delay = filtered_df.groupby('is_peak_season')['is_delayed'].mean().reset_index()
    peak_delay['Sæson'] = peak_delay['is_peak_season'].map({0: 'Normal Sæson', 1: 'Højsæson'})
    peak_delay['is_delayed'] *= 100
    fig3 = px.bar(peak_delay, x='Sæson', y='is_delayed', 
                  labels={'is_delayed': 'Forsinkelse (%)', 'Sæson': ''},
                  text_auto='.1f', color='Sæson')
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    st.subheader("Pakketype (Bulky)")
    bulky_delay = filtered_df.groupby('is_bulky')['is_delayed'].mean().reset_index()
    bulky_delay['Type'] = bulky_delay['is_bulky'].map({0: 'Almindelig (#0)', 1: 'Bulky (#1)'})
    bulky_delay['is_delayed'] *= 100
    fig4 = px.bar(bulky_delay, x='Type', y='is_delayed', 
                  labels={'is_delayed': 'Forsinkelse (%)', 'Type': ''},
                  text_auto='.1f', color='Type')
    st.plotly_chart(fig4, use_container_width=True)

with c5:
    st.subheader("Pakketype (Skrøbelig)")
    fragile_delay = filtered_df.groupby('is_fragile')['is_delayed'].mean().reset_index()
    fragile_delay['Type'] = fragile_delay['is_fragile'].map({0: 'Almindelig (#0)', 1: 'Skrøbelig (#1)'})
    fragile_delay['is_delayed'] *= 100
    fig5 = px.bar(fragile_delay, x='Type', y='is_delayed', 
                  labels={'is_delayed': 'Forsinkelse (%)', 'Type': ''},
                  text_auto='.1f', color='Type')
    st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

# Ruter / Postnummer analyse
st.subheader("Oversigt over ruter per postnummer (Top 30)")

postcode_stats = filtered_df.groupby(['destination_postcode', 'destination_city']).agg(
    Antal_Pakker=('shipment_id', 'count'),
    Forsinkede_Pakker=('is_delayed', 'sum'),
    Gns_Afstand_km=('distance_km', 'mean')
).reset_index()
        
postcode_stats['Forsinkelsesprocent'] = (postcode_stats['Forsinkede_Pakker'] / postcode_stats['Antal_Pakker'] * 100).round(2)
postcode_stats['Gns_Afstand_km'] = postcode_stats['Gns_Afstand_km'].round(1)

# Sort by volume and take top 30
postcode_stats = postcode_stats.sort_values('Antal_Pakker', ascending=False).head(30)
        
# Configure column types for better viewing
st.dataframe(postcode_stats.style.format({
    "Forsinkelsesprocent": "{:.2f}%", 
    "Gns_Afstand_km": "{:.1f} km"
}), use_container_width=True)

st.markdown("---")

# === MASKINLÆRING (FORUDSIGELSER AF FREMTIDIGE PAKKER) ===
st.subheader("ML Forudsigelse: Bliver pakken forsinket?")
#Herunder bruger vi en 'klog algoritme' (Maskinlæring) til at finde mønstre i historikken og forudsige, om en ny pakke bliver forsinket.

# Funktionen herunder "træner" selve algoritmen. 
# Systemet husker træningen (@st.cache_resource), så den ikke skal lære det hele forfra hele tiden.
@st.cache_resource
def train_model(df):
    # 1. Beslut hvilke informationer fra pakken der oftest påvirker tidsplanen (f.eks. afstand og vægt)
    features = ['distance_km', 'package_weight_kg', 'package_volume_liters', 
                'is_bulky', 'is_fragile', 'is_peak_season']
    
    # 2. Opret en kopi af dataen, vi udelukkende bruger som lærebog til algoritmen
    df_model = df.copy()
    
    # 3. Oversæt tekstbeskrivelser (f.eks. transportør-navne) til tal og koder ('one-hot encode'). 
    # Dette skyldes at maskinen kun forstår regnestykker og ikke bogstaver.
    df_model = pd.get_dummies(df_model, columns=['carrier', 'warehouse_city'])
    
    # 4. Slet kolonner med irrelevante detaljer (såsom pakkenumre eller dato-frimærker), 
    # da sådanne tilfældige facts bare vil snyde og drille algoritmen under indlæring.
    cols_to_drop = ['shipment_id', 'order_datetime', 'order_weekday', 'order_hour', 
                    'dispatch_datetime', 'dispatch_weekday', 'dispatch_hour', 
                    'warehouse_id', 'destination_postcode', 'destination_city', 'destination_region',
                    'destination_area_type', 'is_delayed']
    valid_colsToDrop = [c for c in cols_to_drop if c in df_model.columns]
    
    # 5. Her adskiller vi 'facitlisten' (om den er forsinket "is_delayed") fra de oplysninger maskinen må kigge på
    X = df_model.drop(columns=valid_colsToDrop)
    y = df_model['is_delayed']
    
    model_columns = X.columns
    
    # 6. Opdel al data i 2 puljer: En stor pulje (80%) til maskinens daglige læring, og 
    # en ekstra lille pulje (20%) som vi låser inde til sidst og bruger som dens afsluttende eksamen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 7. Start selve hjerne-skolegangen ("Gradient Boosting"). Dette er opskriften som maskinen lærer ud fra.
    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # 8. Lad den nu gætte (forudsige) på de 20% hemmelige testdata ('eksamen') 
    y_pred = clf.predict(X_test)
    
    # 9. Tjek resultaterne og bedøm den for at se hvor god og præcis maskinen i virkeligheden har været
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Send vores "nyudlærte" maskinhjerne samt dens eksamenskarakterer tilbage til hjemmesiden 
    return clf, model_columns, metrics

# Kør endeligt uddannelsesfunktionen, så hjemmesiden får adgang til modellen!
model, model_columns, metrics = train_model(df)

# === BRUGERFLADE: FORMULAR HVOR MAN SELV KAN INDTASTE SIN EGEN PAKKE ===
with st.form("prediction_form"):
    st.write("Indtast detaljer for at lade maskinen vurdere risikoen for forsinkelse:")
    colA, colB, colC = st.columns(3)
    
    with colA:
        p_carrier = st.selectbox("Transportør", df['carrier'].unique())
        p_warehouse = st.selectbox("Distributionscenter", df['warehouse_city'].unique())
        p_distance = st.number_input("Afstand (km)", min_value=0.0, max_value=2000.0, value=50.0)
        
    with colB:
        p_weight = st.number_input("Vægt (kg)", min_value=0.1, max_value=500.0, value=2.5)
        p_volume = st.number_input("Volumen (liter)", min_value=0.1, max_value=5000.0, value=15.0)
        
    with colC:
        p_peak = st.checkbox("Højsæson (Peak Season)?")
        p_bulky = st.checkbox("Er den overdimensioneret (Bulky)?")
        p_fragile = st.checkbox("Er den skrøbelig (Fragile)?")
        
    submitted = st.form_submit_button("Beregn Sandsynlighed")
    
if submitted:
    # Opbyg input dataframe med samme format
    input_dict = {
        'distance_km': [p_distance],
        'package_weight_kg': [p_weight],
        'package_volume_liters': [p_volume],
        'is_bulky': [1 if p_bulky else 0],
        'is_fragile': [1 if p_fragile else 0],
        'is_peak_season': [1 if p_peak else 0],
        f'carrier_{p_carrier}': [1],
        f'warehouse_city_{p_warehouse}': [1]
    }
    
    input_df = pd.DataFrame(input_dict)
    
    # Sørg for at alle forventede kolonner er til stede fra dummies
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    # Sorter kolonner så rækkefølgen matcher modellen
    input_df = input_df[model_columns]
    
    # Fremsig
    prediction = model.predict(input_df)[0]
    prob_delayed = model.predict_proba(input_df)[0][1]
    
    if prediction == 1:
        st.error(f" Modellen forudsiger: **FORSINKET** (Risiko: {prob_delayed*100:.1f}%)")
    else:
        st.success(f" Modellen forudsiger: **RETTIDIG LEVERING** (Risiko for forsinkelse: {prob_delayed*100:.1f}%)")

st.markdown("---")

# === HTS: HVOR GOD ER VORES MODEL HELT PRÆCIST? ===
st.subheader("Modelevaluering (Kvalitetstjek)")
# Her vises maskinens 'afgangsbevis'. Det betyder, hvor præcis modellen reelt var, da vi brugte den til at gætte på den ukendte 20% test-data som vi gemte væk i starten.

# Her oversættes tre vigtige fagudtryk, som beskriver hvor god og følsom algoritmen er til sit job:
# - Accuracy: I hvor stor en procentdel var maskinens svar det helt rigtige (både hvis det drejede sig om at gætte rigtig på forsinkelse OG på tidspunktet)?
# - Precision: Hver eneste gang hjerne-robotten påstod, at pakken var 'Forsinket', hvor tit havde den da faktisk ret i virkeligheden? (Og hvor tit tog den fejl?)
# - Recall: Hvor mange procent af alle *faktiske* forsinkede pakker derude i test-spanden opdagede maskinen egentlig? 
mc1, mc2, mc3 = st.columns(3)
mc1.metric("Nøjagtighed (Accuracy)", f"{metrics['accuracy']*100:.1f}%")
mc2.metric("Præcision (Precision)", f"{metrics['precision']*100:.1f}%")
mc3.metric("Følsomhed (Recall)", f"{metrics['recall']*100:.1f}%")

# Tegner en figur ("Confusion matrix", eller Forvirrings Matrix/Skema på dansk), 
# som detaljeret opgør, i hvilke specifikke tilfælde at maskinen gættede rigtigt og hvornår den snydte og gættede forkert på andres bekostning.
st.markdown("**Forvirrings-skema (Confusion Matrix)**")
fig_cm = px.imshow(metrics['confusion_matrix'], text_auto=True, color_continuous_scale='Blues',
                   labels=dict(x="Hvad maskinen gættede på", y="Hvad forsinkelsen faktisk endte med at blive", color="Antal pakker"),
                   x=['Rettidig (0)', 'Forsinket (1)'], y=['Rettidig (0)', 'Forsinket (1)'])
st.plotly_chart(fig_cm, use_container_width=False)
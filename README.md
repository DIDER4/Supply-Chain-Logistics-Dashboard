# Leveringsanalyse Dashboard

Dette projekt indeholder et Streamlit-baseret dashboard til analyse af logistikdata samt en indbygget Machine Learning-klassifikator til forudsigelse af forsinkede pakker (`is_delayed`).

## Funktioner

* **Dataanalyse:** Visning af forsinkelsesrater aggregeret på features som `carrier`, `warehouse_city`, `is_peak_season`, `is_bulky` og `is_fragile` vha. Plotly bar charts.
* **Interaktiv filtrering:** Filtrering af dataudtræk og visualiseringer baseret på transportør og distributionscenter.
* **Geografisk Analyse:** Aggregeret tabeloversigt over ruter grupperet på destinationens postnummer (`destination_postcode`), sorteret efter forsendelsesvolumen. Inkluderer beregnet forsinkelsesprocent og gennemsnitlig transportafstand.
* **ML Klassifikation:** En præ-trænet `GradientBoostingClassifier` (fra `scikit-learn`) integreret i en Streamlit Form til "on-the-fly" forudsigelse af forsinkelsessandsynligheden på nye forsendelser.
* **Systemevaluering:** Output af klassifikatorens præstation regnet på et 20% test-split, herunder metrics for Accuracy, Precision, Recall samt generering af korresponderende Confusion Matrix pyplot.

## Installation

### Forudsætninger
Sørg for at have et Python 3-miljø konfigureret på maskinen. 

### Trin
1. Åbn din foretrukne terminal og navigér til rodbiblioteket.
2. Installer de nødvendige afhængigheder via den medfølgende konfigurationsfil:
   ```bash
   pip install -r requirements.txt
   ```
3. Eksekver dashboardet via Streamlit CLI:
   ```bash
   streamlit run dashboard.py
   ```
Applikationen vil starte og automatisk åbne `http://localhost:8501` i din standardbrowser.

## Projektstruktur

* `dashboard.py`: Programmets hovedfil (Entrypoint). Inkluderer frontend rendering (Streamlit) og backend data pipeline (Pandas & Scikit-learn).
* `dk_parcel_delivery_basic_.csv`: Det anvendte statiske kildedatasæt, der udnyttes til træning af modellen og rendering af statistik. *Dette er udelukkende dummy data*
* `requirements.txt`: Liste over eksterne pakke-afhængigheder for deployment af projektmiljøet.

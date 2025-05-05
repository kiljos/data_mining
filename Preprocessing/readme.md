# Preprocessing-Modul

## Struktur & Idee

Wir nutzen einen **modularen Ansatz** zur Datenvorverarbeitung:

### 1. `preprocessing_pipeline_initial.py`
- Liest Rohdaten ein
- Führt grundlegende Transformationen durch (z. B. Spalten umbenennen, Datentypen korrigieren, Missing Values, ggf. Outlier)
- **wichtig**: hier wird derzeit noch Offer Description gedroppt, weil es die Modellleistung deutlich verschlechtert hat --> wer also die offer_description pipeline importiert, muss das vorher aus initial rauslöschen
- Gibt einen sauberen `df` zurück

### 2. Weitere Skripte (z. B. `preprocessing_pipeline_segment.py`, `preprocessing_pipeline_offer_description.py`)
- Bauen auf dem Output des initialen Skripts `preprocessing_pipeline_initial.py` auf
- Enthalten transformationsspezifische Schritte (z. B. Segment, Offerdescription)

### 3. `split_data.py`
- Trennt den final verarbeiteten Datensatz in Trainings-, Validierungs- und Testdaten
- hier eventuell auch cross-validation in Zukunft einbauen


---

## Ausführung

Ein Beispielablauf könnte sein:

```python
from Preprocessing.preprocessing_pipeline_initial import preprocessing_pipeline
df = preprocessing_pipeline()
X_train, X_test, y_train, y_test , X,y, categorical_features , numeric_features = split_data(df)

# Dann z. B.:
from Preprocessing.preprocessing_pipeline_initial import preprocessing_pipeline
from Preprocessing.preprocessing_pipeline_segment import preprocessing_pipeline_segment
df = preprocessing_pipeline()
df = preprocessing_pipeline_segment(df)
X_train, X_test, y_train, y_test , X,y, categorical_features , numeric_features = split_data(df)
```
$\rightarrow$ so lassen sich die beiden Ansätze, einmal mit segment und einmal ohne Segment recht schnell vergleichen


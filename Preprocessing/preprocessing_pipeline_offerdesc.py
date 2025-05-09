import pandas as pd

def preprocessing_pipeline_offerdesc(df):

    from sklearn.feature_extraction.text import CountVectorizer

    # Liste von deutschen Stopwörter, die nicht mitgezählt werden sollen (generiert von ChatGPT)
    german_stopwords = [
    "aber", "abgesehen", "alle", "allem", "allen", "alles", "als", "also", "am", "an", "andere", "anderen", "anderm", 
    "anderer", "anderes", "auch", "auf", "aus", "bei", "beide", "beiden", "beim", "beiner", "beim", "bis", "bisschen", 
    "bleiben", "durch", "ein", "eine", "einem", "einen", "einer", "eines", "er", "es", "essen", "etwas", "für", "gegen", 
    "geht", "habe", "haben", "hat", "hatte", "hätten", "hattest", "hattete", "hier", "hin", "hinter", "ich", "ihn", "ihm", 
    "ihr", "ihre", "immer", "in", "indem", "ins", "irgendwas", "ist", "ja", "jeder", "jeden", "jedes", "jedoch", "kann", 
    "kein", "keine", "konnte", "konnte", "machen", "mehr", "mit", "muss", "muss", "nach", "nicht", "nichts", "nur", "ob", 
    "oder", "ohne", "sehr", "sich", "sie", "sind", "so", "sollen", "sollte", "sondern", "um", "und", "uns", "unsere", 
    "vom", "von", "vor", "während", "weil", "wer", "werden", "wie", "wieder", "wieso", "wir", "wird", "wirst", "wo", 
    "woher", "wohin", "zu", "zum", "zur", "zwar", "zwischen"
    ]

    # Entferne Sonderzeichen und Zahlen (weil zu unaussagekräftig) und wandle alles in Kleinbuchstaben um
    # Wird in die neue Spalte _cleaned geschrieben
    df["offer_description_cleaned"] = (
    df["offer_description"]
    .str.replace(r'\d+', '', regex=True)             # Entfernt alle Zahlen
    .str.replace(r'[^\w\s]', '', regex=True)         # Entfernt Sonderzeichen
    .str.lower()                                     # Alles klein
    )

    #Sollten eigentlich schon draußen sein, aber entfernt zur sicherheit nochmal alle leeren werte 
    df = df.dropna(subset=["offer_description_cleaned"]) 

    # Erstellen eines CountVectorizer, der die deutschen Stopwörter verwendet -> nimmt sich hier jetzt die 50 häufigsten wörter
    vectorizer = CountVectorizer(analyzer='word', stop_words=german_stopwords, max_features=50)

    # Anwendung des Vectorizers
    X = vectorizer.fit_transform(df["offer_description_cleaned"])

    # Erstellen eines DataFrame, um die Häufigkeit der Wörter zu sehen
    word_freq = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    # Berechnung der Häufigkeit jedes Schlagworts
    word_count = word_freq.sum().sort_values(ascending=False)

    # Die oben ausgegebene Liste an häufigsten Wörtern wurde dann per Hand aussortiert, e.g. Wörter wie "auto, matic" usw. wurden entfernt
    # Und ähnliche Wörter in Kategorien zusammengefasst z.B. "quattro, xDrive und awd = all wheel drive"
    # Einlesen der Kategorien aus der csv-datei "offer_description_utf8.csv"
    kat = pd.read_csv("offer_description_utf8.csv", sep=";")
    # Werte ohne zugeordnete Kategorien werden gelöscht
    kat = kat.dropna(subset=["Kategorie"])

    # Kategorien + zugehörige Abkürzungen in Dictionary gruppieren
    abkuerzung_dict = kat.groupby("Kategorie")["Abkuerzung"].apply(list).to_dict()

    # Für jede Kategorie eine neue Spalte erzeugen, die boolesche Werte enthält
    # Also z.B. has_navigation? yes / no
    for kategorie, abkuerzungen in abkuerzung_dict.items():
        df[kategorie] = df["offer_description"].apply(
        lambda text: True if any(abk.lower() in text.lower() for abk in abkuerzungen) else False
        )

    df = df.drop(['offer_description'], axis=1)

    return df
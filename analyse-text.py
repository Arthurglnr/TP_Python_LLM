import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

# Chargement du modèle spaCy français
nlp = spacy.load("fr_core_news_sm")

def analyse_frequence(mots):
    """Étape 3.1 : Compter les mots les plus fréquents"""
    compteur = Counter(mots)
    print("\n🔢 Mots les plus fréquents :")
    for mot, freq in compteur.most_common(10):
        print(f"{mot} : {freq}")

    return compteur

def generer_wordcloud(compteur):
    """Visualisation avec un nuage de mots"""
    wordcloud = WordCloud(width=800, height=400, background_color='white')
    wordcloud.generate_from_frequencies(compteur)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Nuage de mots")
    plt.show()

def analyse_tfidf(listes_mots):
    """TF-IDF même si un seul document"""
    print("\n📊 Analyse TF-IDF")

    documents = [" ".join(mots) for mots in listes_mots]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    features = vectorizer.get_feature_names_out()
    for i, doc in enumerate(documents):
        print(f"\n📄 Document {i + 1} :")
        tfidf_scores = tfidf_matrix[i].tocoo()
        sorted_items = sorted(zip(tfidf_scores.col, tfidf_scores.data), key=lambda x: x[1], reverse=True)
        for idx, score in sorted_items[:10]:
            print(f"{features[idx]} : {score:.4f}")


def analyse_ner(mots):
    """Étape 3.3 : Détection des entités nommées"""
    print("\n🧠 Entités nommées détectées (NER) :")
    texte = " ".join(mots)
    doc = nlp(texte)

    entites = [(ent.text, ent.label_) for ent in doc.ents]
    for ent in entites:
        print(f"{ent[0]} → {ent[1]}")

    # Optionnel : Regrouper par type
    personnes = [ent.text for ent in doc.ents if ent.label_ == "PER"]
    lieux = [ent.text for ent in doc.ents if ent.label_ == "LOC"]
    organisations = [ent.text for ent in doc.ents if ent.label_ == "ORG"]

    print("\n📍 Résumé des entités :")
    print("Personnes :", personnes)
    print("Lieux     :", lieux)
    print("Organisations :", organisations)

def analyser_liste_mots(liste_mots, autres_textes=None):
    """
    Fonction principale d’analyse
    - liste_mots : liste de mots du texte principal
    - autres_textes : liste de listes de mots à comparer [liste1, liste2, ...]
    """
    # Étape 3.1 - Fréquences
    compteur = analyse_frequence(liste_mots)
    generer_wordcloud(compteur)

    # Étape 3.2 - TF-IDF (comparaison)
    if autres_textes:
        analyse_tfidf([liste_mots] + autres_textes)
    else:
        analyse_tfidf([liste_mots])

    # Étape 3.3 - NER
    analyse_ner(liste_mots)


if __name__ == "__main__":
    # Exemple de texte principal sous forme de liste de mots
    liste_mots = [
        'emmanuel', 'macron', 'decembre', 'amien', 'homme', 'etat', 'francai', 'presider', 'republique', 'mai', 'parcours', 'marque', 'ascension', 'rapide', 'allier', 'experience', 'fonction', 'public', 'secteur', 'prive', 'politique', 'formation', 'debut', 'professionnel', 'etude', 'philosophie', 'universite', 'pari', 'nanterr', 'emmanuel', 'macron', 'integre', 'institut', 'etude', 'politique', 'pari', 'science', 'po', 'ecole', 'national', 'administration', 'ener', 'sortir', 'diplome', 'debute', 'carriere', 'qu', 'inspecteur', 'finance', 'rejoindre', 'banque', 'affaire', 'rothschild', 'cie', 'devenir', 'associ', 'gerer', 'parcour', 'politique', 'emmanuel', 'macron', 'nomm', 'secretair', 'general', 'adjoint', 'presidence', 'republique', 'francoi', 'hollande', 'an', 'tard', 'devenir', 'ministre', 'economie', 'industrie', 'numerique', 'gouvernement', 'manuel', 'valls', 'poste', 'qu', 'occuper', 'mandat', 'promouvoir', 'reforme', 'viser', 'liberaliser', 'economie', 'francais', 'presidence', 'republique', 'avril', 'emmanuel', 'macron', 'fondre', 'mouvement', 'politique', 'marche', 'annonce', 'candidature', 'election', 'presidentiell', 'elu', 'mai', 'devenir', 'jeune', 'president', 'republique', 'francais', 'mandat', 'marque', 'reforme', 'economique', 'social', 'defi', 'manifestation', 'gilet', 'jaune', 'gestion', 'pandemie', 'covid', 'reelu', 'presider', 'francai', 'decennie', 'obtenir', 'second', 'mandat', 'defi', 'critique', 'reelection', 'emmanuel', 'macron', 'face', 'instabilit', 'politique', 'accroître', 'decision', 'convoquer', 'election', 'legislative', 'anticipees', 'conduire', 'parlement', 'fragment', 'compliquer', 'formation', 'gouvernement', 'stable', 'situation', 'entraine', 'critique', 'gestion', 'pouvoir', 'alimenter', 'debat', 'style', 'leadership', 'vie', 'prive', 'emmanuel', 'macron', 'marier', 'brigitte', 'trogneu', 'ancien', 'professeure', 'lettre', 'an', 'ainee', 'union', 'bien', 'mediatisee', 'temoign', 'relation', 'solide', 'complice'
    ]

    # Appel de la fonction principale d’analyse (sans autres textes)
    analyser_liste_mots(liste_mots)

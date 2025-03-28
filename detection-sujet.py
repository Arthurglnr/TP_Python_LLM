from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import spacy
import numpy as np

# Chargement du modèle spaCy
nlp = spacy.load("fr_core_news_sm")

# Seuil de mots-clés associés à un thème connu pour être classé (ex: si < 2 → inconnu)
SEUIL_SIMILARITE = 2

# Thèmes existants (doivent correspondre à ceux du fichier classification_texte.py)
THEMES = {
    "science": ["recherche", "univers", "physique", "biologie", "experience", "atome", "adn"],
    "politique": ["gouvernement", "president", "ministre", "loi", "parlement", "république"],
    "sport": ["match", "joueur", "football", "score", "tennis", "stade", "but"],
    "économie": ["marché", "bourse", "finance", "banque", "argent"],
    "technologie": ["intelligence", "artificielle", "algorithme", "réseau", "robot", "ordinateur"],
}

def detecter_theme(mots):
    """Détecte si le texte correspond à un thème connu"""
    scores = {theme: 0 for theme in THEMES}
    for mot in mots:
        for theme, keywords in THEMES.items():
            if mot in keywords:
                scores[theme] += 1
    theme_dominant = max(scores, key=scores.get)
    if scores[theme_dominant] < SEUIL_SIMILARITE:
        return "Inconnu"
    return theme_dominant

def transformer_textes(textes_liste_mots):
    """Transforme les textes (liste de mots) en textes string pour TF-IDF"""
    return [" ".join(mots) for mots in textes_liste_mots]

def clusteriser_textes(textes_liste_mots, n_clusters=2):
    """Étape 5.2 : Clustering des textes 'Inconnus' pour regrouper des sujets similaires"""
    textes = transformer_textes(textes_liste_mots)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(textes)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(textes_liste_mots[i])

    return clusters

def generer_nom_theme(cluster):
    """Étape 5.3 : Générer un nom pour un nouveau thème à partir du cluster"""
    mots = [mot for doc in cluster for mot in doc]
    compteur = Counter(mots)
    termes_dominants = [mot for mot, freq in compteur.most_common(3)]

    # Éviter les doublons de thèmes existants
    for theme, mots_cle in THEMES.items():
        if any(mot in mots_cle for mot in termes_dominants):
            return f"Nouveau_{theme}"

    return "_".join(termes_dominants).capitalize()

def traiter_textes(textes_liste_mots):
    if not textes_liste_mots or not isinstance(textes_liste_mots[0], list):
        raise ValueError("Le paramètre doit être une liste de textes, chaque texte étant une liste de mots.")

    """Pipeline complet : filtrer les inconnus, clusteriser et nommer les nouveaux thèmes"""
    inconnus = []
    for texte in textes_liste_mots:
        if detecter_theme(texte) == "Inconnu":
            inconnus.append(texte)

    if not inconnus:
        print("✅ Aucun texte inconnu à traiter.")
        return

    print(f"🔎 {len(inconnus)} texte(s) non classés détectés.")

    clusters = clusteriser_textes(inconnus, n_clusters=2)

    for label, cluster in clusters.items():
        nom_theme = generer_nom_theme(cluster)
        print(f"\n🆕 Nouveau thème détecté : {nom_theme}")
        print("Exemples de textes :")
        for texte in cluster[:2]:
            print(" -", " ".join(texte[:10]), "...")


if __name__ == "__main__":
    texte_macron = [
        'emmanuel', 'macron', 'decembre', 'amien', 'homme', 'etat', 'francai', 'presider', 'republique', 'mai', 'parcours', 'marque', 'ascension', 'rapide', 'allier', 'experience', 'fonction', 'public', 'secteur', 'prive', 'politique', 'formation', 'debut', 'professionnel', 'etude', 'philosophie', 'universite', 'pari', 'nanterr', 'emmanuel', 'macron', 'integre', 'institut', 'etude', 'politique', 'pari', 'science', 'po', 'ecole', 'national', 'administration', 'ener', 'sortir', 'diplome', 'debute', 'carriere', 'qu', 'inspecteur', 'finance', 'rejoindre', 'banque', 'affaire', 'rothschild', 'cie', 'devenir', 'associ', 'gerer', 'parcour', 'politique', 'emmanuel', 'macron', 'nomm', 'secretair', 'general', 'adjoint', 'presidence', 'republique', 'francoi', 'hollande', 'an', 'tard', 'devenir', 'ministre', 'economie', 'industrie', 'numerique', 'gouvernement', 'manuel', 'valls', 'poste', 'qu', 'occuper', 'mandat', 'promouvoir', 'reforme', 'viser', 'liberaliser', 'economie', 'francais', 'presidence', 'republique', 'avril', 'emmanuel', 'macron', 'fondre', 'mouvement', 'politique', 'marche', 'annonce', 'candidature', 'election', 'presidentiell', 'elu', 'mai', 'devenir', 'jeune', 'president', 'republique', 'francais', 'mandat', 'marque', 'reforme', 'economique', 'social', 'defi', 'manifestation', 'gilet', 'jaune', 'gestion', 'pandemie', 'covid', 'reelu', 'presider', 'francai', 'decennie', 'obtenir', 'second', 'mandat', 'defi', 'critique', 'reelection', 'emmanuel', 'macron', 'face', 'instabilit', 'politique', 'accroître', 'decision', 'convoquer', 'election', 'legislative', 'anticipees', 'conduire', 'parlement', 'fragment', 'compliquer', 'formation', 'gouvernement', 'stable', 'situation', 'entraine', 'critique', 'gestion', 'pouvoir', 'alimenter', 'debat', 'style', 'leadership', 'vie', 'prive', 'emmanuel', 'macron', 'marier', 'brigitte', 'trogneu', 'ancien', 'professeure', 'lettre', 'an', 'ainee', 'union', 'bien', 'mediatisee', 'temoign', 'relation', 'solide', 'complice'
    ]

    # Appelle ta fonction principale de classification
    traiter_textes([texte_macron])
    

# ne marche pas
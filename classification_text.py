import spacy
from collections import Counter

# Étape 4.1 : Thèmes prédéfinis avec mots-clés
THEMES = {
    "science": ["recherche", "univers", "physique", "biologie", "experience", "atome", "adn", "intelligence", "algorithme"],
    "politique": ["gouvernement", "president", "ministre", "loi", "parlement", "république", "parti", "vote"],
    "sport": ["match", "joueur", "football", "score", "olympique", "tennis", "stade", "but", "compétition"],
    "économie": ["marché", "bourse", "argent", "finance", "banque", "PIB", "inflation", "chômage", "entreprise"],
    "technologie": ["intelligence", "artificielle", "algorithme", "ordinateur", "réseau", "digital", "robot", "code"],
    "santé": ["virus", "vaccin", "hopital", "maladie", "symptôme", "soin", "traitement", "pandémie"],
}

# Charger le modèle spaCy français
nlp = spacy.load("fr_core_news_sm")

def detecter_theme_par_motcles(mots):
    """Étape 4.2 : Classifieur simple basé sur mots-clés"""
    scores = {theme: 0 for theme in THEMES}
    for mot in mots:
        for theme, keywords in THEMES.items():
            if mot in keywords:
                scores[theme] += 1

    theme_dominant = max(scores, key=scores.get)
    if scores[theme_dominant] == 0:
        return "Inconnu"
    return theme_dominant

def detecter_theme_par_modele(texte):
    """Étape 4.3 : Classification NLP simple avec spaCy"""
    doc = nlp(texte)
    # On extrait les mots les plus fréquents dans le texte
    mots = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    compteur = Counter(mots)
    print("\n📊 Mots dominants dans le texte (via NLP) :")
    for mot, freq in compteur.most_common(10):
        print(f"{mot} : {freq}")
    return detecter_theme_par_motcles(mots)

def classer_texte(liste_mots):
    """Classe un texte selon les mots et NLP"""
    print("\n🧠 Classification par mots-clés :")
    theme1 = detecter_theme_par_motcles(liste_mots)
    print("→ Thème détecté :", theme1)

    print("\n🔍 Classification automatique via modèle NLP :")
    texte = " ".join(liste_mots)
    theme2 = detecter_theme_par_modele(texte)
    print("→ Thème NLP :", theme2)

    return theme1, theme2

# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple : un texte sur l'intelligence artificielle
    texte_exemple = [
        'emmanuel', 'macron', 'decembre', 'amien', 'homme', 'etat', 'francai', 'presider', 'republique', 'mai', 'parcours', 'marque', 'ascension', 'rapide', 'allier', 'experience', 'fonction', 'public', 'secteur', 'prive', 'politique', 'formation', 'debut', 'professionnel', 'etude', 'philosophie', 'universite', 'pari', 'nanterr', 'emmanuel', 'macron', 'integre', 'institut', 'etude', 'politique', 'pari', 'science', 'po', 'ecole', 'national', 'administration', 'ener', 'sortir', 'diplome', 'debute', 'carriere', 'qu', 'inspecteur', 'finance', 'rejoindre', 'banque', 'affaire', 'rothschild', 'cie', 'devenir', 'associ', 'gerer', 'parcour', 'politique', 'emmanuel', 'macron', 'nomm', 'secretair', 'general', 'adjoint', 'presidence', 'republique', 'francoi', 'hollande', 'an', 'tard', 'devenir', 'ministre', 'economie', 'industrie', 'numerique', 'gouvernement', 'manuel', 'valls', 'poste', 'qu', 'occuper', 'mandat', 'promouvoir', 'reforme', 'viser', 'liberaliser', 'economie', 'francais', 'presidence', 'republique', 'avril', 'emmanuel', 'macron', 'fondre', 'mouvement', 'politique', 'marche', 'annonce', 'candidature', 'election', 'presidentiell', 'elu', 'mai', 'devenir', 'jeune', 'president', 'republique', 'francais', 'mandat', 'marque', 'reforme', 'economique', 'social', 'defi', 'manifestation', 'gilet', 'jaune', 'gestion', 'pandemie', 'covid', 'reelu', 'presider', 'francai', 'decennie', 'obtenir', 'second', 'mandat', 'defi', 'critique', 'reelection', 'emmanuel', 'macron', 'face', 'instabilit', 'politique', 'accroître', 'decision', 'convoquer', 'election', 'legislative', 'anticipees', 'conduire', 'parlement', 'fragment', 'compliquer', 'formation', 'gouvernement', 'stable', 'situation', 'entraine', 'critique', 'gestion', 'pouvoir', 'alimenter', 'debat', 'style', 'leadership', 'vie', 'prive', 'emmanuel', 'macron', 'marier', 'brigitte', 'trogneu', 'ancien', 'professeure', 'lettre', 'an', 'ainee', 'union', 'bien', 'mediatisee', 'temoign', 'relation', 'solide', 'complice'
    ]

    classer_texte(texte_exemple)

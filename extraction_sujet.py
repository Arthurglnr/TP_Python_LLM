import spacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from collections import Counter

# Charger le modèle spaCy français
nlp = spacy.load("fr_core_news_sm")

# Thèmes déjà définis (mêmes que classification_texte.py)
THEMES = {
    "science": ["recherche", "univers", "physique", "biologie", "experience"],
    "politique": ["gouvernement", "president", "loi", "parlement", "hollande", "macron"],
    "sport": ["match", "joueur", "football", "tennis", "but"],
    "économie": ["argent", "finance", "banque", "economie", "bourse"],
    "technologie": ["intelligence", "algorithme", "ordinateur", "digital", "robot"],
}

def detecter_theme_par_motcles(mots):
    scores = {theme: 0 for theme in THEMES}
    for mot in mots:
        for theme, keywords in THEMES.items():
            if mot in keywords:
                scores[theme] += 1
    theme_dominant = max(scores, key=scores.get)
    return theme_dominant if scores[theme_dominant] > 0 else "Inconnu"

def identifier_phrase_cle(texte):
    """Étape 6.1 : Trouver la phrase contenant les termes importants"""
    doc = nlp(texte)
    mots_importants = [token.lemma_ for token in doc if token.pos_ in {"NOUN", "PROPN", "VERB"} and not token.is_stop]
    compteur = Counter(mots_importants)

    phrase_cle = ""
    score_max = 0
    for sent in doc.sents:
        mots_sentence = [token.lemma_ for token in sent if token.lemma_ in compteur]
        score = sum(compteur[mot] for mot in mots_sentence)
        if score > score_max:
            score_max = score
            phrase_cle = sent.text

    return phrase_cle.strip()

def resumer_texte(texte, nb_phrases=1):
    """Étape 6.2 : Résumer avec TextRank (Sumy)"""
    parser = PlaintextParser.from_string(texte, Tokenizer("french"))
    summarizer = TextRankSummarizer()
    resultats = summarizer(parser.document, nb_phrases)
    return " ".join(str(phrase) for phrase in resultats)

def comparer_theme_et_sujet(mots, phrase_sujet, theme_detecte):
    """Étape 6.3 : Comparer le sujet extrait et le thème classé"""
    doc = nlp(phrase_sujet)
    mots_sujet = [token.lemma_ for token in doc if not token.is_stop]
    sujet_theme = detecter_theme_par_motcles(mots_sujet)

    print("\n🔍 Comparaison du thème détecté avec le sujet extrait :")
    print(f"- Thème classé via mots-clés : {theme_detecte}")
    print(f"- Thème détecté via phrase sujet : {sujet_theme}")

    if theme_detecte == sujet_theme:
        print("✅ Sujet et thème cohérents.")
    else:
        print("⚠️ Sujet et thème ne correspondent pas totalement.")

def analyser_sujet(liste_mots):
    """Pipeline principal"""
    texte = " ".join(liste_mots)

    # Étape 6.1
    phrase_cle = identifier_phrase_cle(texte)
    print("\n🧠 Phrase clé identifiée :", phrase_cle)

    # Étape 6.2
    resume = resumer_texte(texte)
    print("\n✍️ Résumé TextRank :", resume)

    # Étape 6.3
    theme = detecter_theme_par_motcles(liste_mots)
    comparer_theme_et_sujet(liste_mots, phrase_cle, theme)

if __name__ == "__main__":
    texte_macron = [
        'emmanuel', 'macron', 'decembre', 'amien', 'homme', 'etat', 'francai', 'presider', 'republique', 'mai', 'parcours', 'marque', 'ascension', 'rapide', 'allier', 'experience', 'fonction', 'public', 'secteur', 'prive', 'politique', 'formation', 'debut', 'professionnel', 'etude', 'philosophie', 'universite', 'pari', 'nanterr', 'emmanuel', 'macron', 'integre', 'institut', 'etude', 'politique', 'pari', 'science', 'po', 'ecole', 'national', 'administration', 'ener', 'sortir', 'diplome', 'debute', 'carriere', 'qu', 'inspecteur', 'finance', 'rejoindre', 'banque', 'affaire', 'rothschild', 'cie', 'devenir', 'associ', 'gerer', 'parcour', 'politique', 'emmanuel', 'macron', 'nomm', 'secretair', 'general', 'adjoint', 'presidence', 'republique', 'francoi', 'hollande', 'an', 'tard', 'devenir', 'ministre', 'economie', 'industrie', 'numerique', 'gouvernement', 'manuel', 'valls', 'poste', 'qu', 'occuper', 'mandat', 'promouvoir', 'reforme', 'viser', 'liberaliser', 'economie', 'francais', 'presidence', 'republique', 'avril', 'emmanuel', 'macron', 'fondre', 'mouvement', 'politique', 'marche', 'annonce', 'candidature', 'election', 'presidentiell', 'elu', 'mai', 'devenir', 'jeune', 'president', 'republique', 'francais', 'mandat', 'marque', 'reforme', 'economique', 'social', 'defi', 'manifestation', 'gilet', 'jaune', 'gestion', 'pandemie', 'covid', 'reelu', 'presider', 'francai', 'decennie', 'obtenir', 'second', 'mandat', 'defi', 'critique', 'reelection', 'emmanuel', 'macron', 'face', 'instabilit', 'politique', 'accroître', 'decision', 'convoquer', 'election', 'legislative', 'anticipees', 'conduire', 'parlement', 'fragment', 'compliquer', 'formation', 'gouvernement', 'stable', 'situation', 'entraine', 'critique', 'gestion', 'pouvoir', 'alimenter', 'debat', 'style', 'leadership', 'vie', 'prive', 'emmanuel', 'macron', 'marier', 'brigitte', 'trogneu', 'ancien', 'professeure', 'lettre', 'an', 'ainee', 'union', 'bien', 'mediatisee', 'temoign', 'relation', 'solide', 'complice'
    ]

    analyser_sujet(texte_macron)

# ne marche pas
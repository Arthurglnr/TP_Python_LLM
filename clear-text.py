import spacy
import os
import unicodedata
import re

# Chargement du modèle spaCy français
try:
    nlp = spacy.load("fr_core_news_sm")
except:
    print("Erreur : le modèle spaCy 'fr_core_news_sm' n'est pas installé.")
    exit()

# Liste personnalisée de mots à supprimer
mots_a_supprimer = {
    "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux", "en",
    "l", "d", "ce", "ces", "cet", "cette", "mon", "ton", "son", "ma", "ta",
    "sa", "mes", "tes", "ses", "nos", "vos", "leurs"
}

def enlever_accents(texte):
    """Remplacer les caractères accentués par leurs équivalents sans accent"""
    texte = unicodedata.normalize('NFD', texte)
    texte = texte.encode('ascii', 'ignore').decode('utf-8')
    return texte

def charger_fichier(chemin):
    """Étape 2.1 : Charger un fichier texte et afficher les premières lignes"""
    if not os.path.exists(chemin):
        print(f"Erreur : le fichier '{chemin}' n'existe pas.")
        return None
    
    try:
        with open(chemin, 'r', encoding='utf-8') as f:
            contenu = f.read()
            print("📄 Premières lignes du fichier :")
            print("\n".join(contenu.splitlines()[:5]))
            return contenu
    except UnicodeDecodeError:
        print("Erreur : problème d'encodage du fichier.")
        return None
    except Exception as e:
        print(f"Erreur inconnue : {e}")
        return None

def nettoyer_texte(texte):
    """Étape 2.2 : Nettoyage et préparation du texte"""
    print("\n🔍 Texte original (extrait) :")
    print(texte[:300], "...\n")

    # 1. Mise en minuscules
    texte = texte.lower()

    # 2. Remplacement des caractères accentués
    texte = enlever_accents(texte)

    # 3. Suppression de tout sauf lettres et espaces
    texte = re.sub(r"[^a-z\s]", " ", texte)

    # 4. Tokenisation
    doc = nlp(texte)

    tokens_nettoyes = []
    for token in doc:
        if (
            not token.is_stop
            and token.text not in mots_a_supprimer
            and not token.is_punct
            and not token.is_space
        ):
            tokens_nettoyes.append(token.lemma_)

    print("✅ Liste finale des mots nettoyés :")
    print(tokens_nettoyes)
    return tokens_nettoyes

# 🔧 Exemple d'utilisation
if __name__ == "__main__":
    chemin_fichier = "mon_texte.txt"  # Remplace ce chemin par ton fichier réel
    texte = charger_fichier(chemin_fichier)
    
    if texte:
        nettoyer_texte(texte)

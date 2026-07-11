# 🏥 Diabetes Risk Prediction & Clinical Explainability

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikit-learn)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

Projet d'apprentissage automatique (**Machine Learning**) orienté vers l'**interprétabilité clinique et pédagogique**.  
L'objectif est de prédire le risque de diabète à partir de données médicales (Pima Indians Diabetes Dataset), de comparer un modèle linéaire (*Régression Logistique*) à un modèle non-linéaire (*Forêt Aléatoire*), et de décrypter les prédictions du modèle ensembliste à l'aide de **SHAP** (*SHapley Additive exPlanations*).

---

## 🎯 Objectif du Projet & Enjeu Clinique : Comprendre les Métriques

Dans un projet d'apprentissage automatique appliqué à la santé, l'évaluation d'un modèle ne peut pas se résumer à son taux de bonne classification global (*Accuracy*). Il est indispensable d'adapter le choix des métriques aux réalités du diagnostic médical et aux conséquences pour les patients.

### 📘 Définition claire des métriques d'évaluation :

1. **Accuracy (Exactitude globale)** : *« Sur l'ensemble de la population examinée (sains et malades), quelle proportion de patients a été classée correctement ? »*  
   - Elle mesure la performance globale du modèle sur toutes les classes.  
   - ⚠️ **Limite en contexte médical** : Si une pathologie ne touche que 1 % d'une population, un modèle qui prédirait systématiquement l'absence de maladie (*zéro détection*) atteindrait **99 % d'Accuracy**. Pourtant, sa valeur clinique serait nulle puisqu'il n'identifierait aucun patient malade. L'Accuracy seule est donc insuffisante face à des classes déséquilibrées.

2. **Recall (Sensibilité / Taux de détection)** : *« Parmi tous les patients qui sont réellement atteints de diabète, quelle proportion le modèle parvient-il à identifier ? »*  
   - Il évalue la capacité de l'algorithme à ne laisser passer aucun cas positif. Un *Recall* élevé garantit que la grande majorité des patients malades sont repérés et pris en charge à temps.

3. **Precision (Valeur Prédictive Positive)** : *« Lorsque le modèle prédit qu'un patient est à risque de diabète, quelle est la probabilité que ce diagnostic soit exact ? »*  
   - Elle évalue la fiabilité des alertes déclenchées par le modèle. Par exemple, une *Precision* de 70 % signifie que sur 100 patients identifiés comme étant à risque par l'algorithme, 70 le sont réellement (les 30 autres constituant des diagnostics faussement positifs).

---

> [!IMPORTANT]
> **Le dilemme médical en situation de dépistage (Recall vs Precision)** :  
> - **Le Faux Négatif (déficit de Recall)** : Un patient réellement diabétique n'est pas détecté par le modèle. Il ne reçoit pas d'information ni de prise en charge thérapeutique précoce, ce qui l'expose à des complications cliniques sévères (atteintes rénales, cardiovasculaires ou rétiniennes).  
> - **Le Faux Positif (déficit de Precision)** : Un patient sain est identifié à tort comme suspect. Il devra subir une consultation médicale et des examens complémentaires (par exemple une prise de sang de contrôle) pour écarter le diagnostic.  
> 
> 👉 **En santé publique et lors d'une phase de dépistage, ne pas détecter un patient malade (Faux Négatif) a des conséquences de santé bien plus graves qu'une investigation complémentaire chez un patient sain (Faux Positif). La priorité clinique absolue est donc de maximiser le RECALL (Sensibilité), quitte à tolérer un taux légèrement plus élevé de fausses alertes.**

---

## 📊 Dataset & Défi des "Zéros Masqués"

- **Source** : [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (Kaggle / UCI Machine Learning Repository).
- **Volume** : 768 patientes (femmes de la communauté Pima, ≥ 21 ans), 8 variables cliniques et 1 variable cible binaire (`Outcome`: 0 = non-diabétique, 1 = diabétique).
- **Problématique majeure identifiée** : De nombreuses mesures vitales contiennent des `0` qui sont biologiquement impossibles :
  - `Glucose = 0` (5 observations)
  - `BloodPressure = 0` (35 observations)
  - `SkinThickness = 0` (227 observations)
  - `Insulin = 0` (374 observations, soit près de 49 % !)
  - `BMI = 0` (11 observations)  
  
  *Traitement apporté* : Ces zéros ont été reconnus comme des **valeurs manquantes déguisées** (`np.nan`), puis imputés intelligemment à l'aide de la **médiane** calculée **uniquement sur le jeu d'entraînement** (pour éliminer tout risque de *Data Leakage*).

---

## 🔬 Méthodologie & Pipeline Pédagogique

Le projet est divisé en 3 notebooks séquentiels, conçus pour être compréhensibles par un public non-initié :

1. **`01_eda.ipynb` (Exploration des données)** : Analyse visuelle des distributions, de l'asymétrie des classes (65% vs 35%) et des corrélations cliniques.
2. **`02_preprocessing.ipynb` (Prétraitement & Data Leakage)** : Découpage stratifié Train/Test ($80\% / 20\%$), imputation par la médiane et standardisation (`StandardScaler`) en veillant à ne jamais exposer les données de test à l'étape d'apprentissage.
3. **`03_modeling.ipynb` (Modélisation & Interprétabilité)** : Entraînement, comparaison et dissection des modèles via des visualisations 2D des frontières de décision, l'inspection d'un arbre de décision individuel et une analyse de contribution SHAP.

---

## 📈 Comparaison & Résultats des Modèles

Les deux algorithmes ont été évalués sur l'échantillon de test (`X_test`, 154 patientes dont 54 diabétiques) :

| Modèle | Accuracy | Precision | Recall (Sensibilité) ⭐ | F1-score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Baseline linéaire) | `70.78 %` | `60.00 %` | `50.00 %` | `54.55 %` | `0.8130` |
| **Random Forest** (Forêt Aléatoire non-linéaire) | **`77.27 %`** | **`70.21 %`** | **`61.11 %`** | **`65.35 %`** | **`0.8181`** |

### 🔍 Analyse clinique des résultats :
- **Le saut de Recall (+11.1 %)** : Sur les 54 patientes diabétiques du jeu de test, la Régression Logistique en détecte 27 (*Recall = 50%*), tandis que la Forêt Aléatoire en détecte 33 (*Recall = 61.1%*).  
  **Le modèle ensembliste permet donc de sauver 6 patients supplémentaires de l'errance médicale sur un simple échantillon de 154 personnes !**
- **Capacité de discrimination (ROC-AUC)** : Avec un score `ROC-AUC ~ 0.82`, les deux modèles montrent une excellente capacité à hiérarchiser les probabilités de risque.

---

## 🧠 Interprétabilité du Modèle (SHAP)

Pour briser l'effet "boîte noire" de la Forêt Aléatoire et inspirer confiance aux praticiens de santé, nous avons utilisé la méthode **SHAP** basée sur la théorie des jeux :

1. **`Glucose` (Taux de sucre sanguin)** : De loin le facteur clinique le plus déterminant. Des valeurs élevées de glucose poussent fortement la prédiction vers la classe positive (diabète).
2. **`BMI` (Indice de Masse Corporelle)** : Le deuxième facteur de risque majeur. Un surpoids/obésité contribue nettement à l'augmentation du score de risque.
3. **`Age` & `DiabetesPedigreeFunction`** : L'âge avancé et les antécédents familiaux agissent comme des facteurs aggravants qui viennent consolider le diagnostic sur des cas limites.

---

## 🛠️ Installation & Exécution locale

```bash
# 1. Cloner le dépôt
git clone https://github.com/Orpheus76/diabetes-risk-prediction.git
cd diabetes-risk-prediction

# 2. Créer et activer l'environnement virtuel Python
python3 -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer Jupyter Notebook pour explorer les analyses
jupyter notebook
```

---

## 📁 Structure du projet

```text
diabetes-risk-prediction/
├── conftest.py                # Configuration des tests pytest
├── data/
│   └── raw/diabetes.csv       # Dataset brut (Pima Indians)
├── models/                    # Modèles entraînés et exportés (.pkl)
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── scaler.pkl
├── notebooks/                 # Notebooks pédagogiques et interactifs
│   ├── 01_eda.ipynb           # Analyse Exploratoire des Données
│   ├── 02_preprocessing.ipynb # Prétraitement rigoureux & Anti-leakage
│   └── 03_modeling.ipynb      # Modélisation, Frontières de décision & SHAP
├── src/                       # Code source réutilisable et modularisé
│   ├── data_loader.py         # Chargement des données
│   ├── preprocessing.py       # Pipeline d'imputation et de standardisation
│   └── train.py               # Fonctions d'apprentissage et d'évaluation
└── tests/                     # Suite de tests unitaires automatisés
    ├── test_preprocessing.py
    └── test_train.py
```

---

## 🚀 Pistes d’amélioration futures

- [ ] **Optimisation du Recall via les seuils de probabilité** : Abaisser le seuil de décision (ex: passer de `0.50` à `0.35`) pour capturer un maximum de cas à risque (*Recall > 85%*).
- [ ] **Gestion du déséquilibre des classes** : Expérimenter **SMOTE** (*Synthetic Minority Over-sampling Technique*) ou l'ajustement des poids (`class_weight='balanced'`).
- [ ] **Modèles de Gradient Boosting** : Évaluer et comparer **XGBoost** et **LightGBM**.
- [ ] **Interface Web Interactive (Streamlit)** : Développer un tableau de bord clinique interactif pour tester le modèle en temps réel.

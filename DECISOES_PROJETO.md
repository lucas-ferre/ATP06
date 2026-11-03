# Decisões de Projeto e Comparação Experimental de Modelos (ATP06)

**Disciplina:** Engenharia de Sistemas Inteligentes (CK0444 – 2026.2)  
**Departamento:** Departamento de Computação (DC) / Universidade Federal do Ceará (UFC)  
**Autor:** Lucas  
**Projeto:** PDS (`pds-model` — Pipeline de Modelos)  

---

## 1. Contextualização e Objetivos da Prática

Na **ATP05**, a investigação debruçou-se sobre a **Pipeline de Dados** (`pds-data`). O diagnóstico exploratório comprovou que o dataset original (`v2_3_1_trf_dataset.csv`) continha **73,04% de arquivos irrelevantes** (testes unitários e scripts auxiliares de build), que diluíam a incidência real de defeitos e enviesavam as métricas de software. A remodelagem aplicada em `tansform_raw_dataset` e a seleção da janela temporal ótima ($\delta=1$, correspondente à release `v3.0.6`) produziram um dataset de alta fidelidade (`v3_0_6_trf_dataset.csv`, com 230 módulos reais de produção e 45,22% de taxa de defeito).

Na presente **ATP06**, o foco transferiu-se para a **Pipeline de Modelos** (`pds-model`), com duas metas fundamentais:
1. **Remodelagem do Setup Experimental e Rigor Metodológico:** Expandir o conjunto de classificadores avaliados em benchmark, incorporando `sklearn.svm.SVC` (`SVC`) e `GradientBoostingClassifier` (`GBC`) ao lado de `LogisticRegression` (`LRC`) e `RandomForestClassifier` (`RFC`), além de corrigir problemas metodológicos de cálculo de métricas probabilísticas, vazamento de dados no `SMOTE` e reprodutibilidade.
2. **Estudo Comparativo Rigoroso:** Avaliar formalmente se a incorporação dos novos algoritmos e a substituição do dataset baseline ruidoso (`v2_3_1_trf_dataset.csv`) pelo dataset tratado da ATP05 (`v3_0_6_trf_dataset.csv`) resulta em um modelo final de melhor performance, menor erro de calibração (`Brier Score`) e maior poder de generalização (`ROC-AUC`).

---

## 2. Melhorias Metodológicas e de Engenharia em `pipeline.py`

### 2.1. Inclusão dos Novos Classificadores e Uso de `sklearn.svm.SVC`
* Na função `start(...)`, a lista de modelos avaliados em `do_benchmark(...)` foi expandida de `["LRC", "RFC"]` para `["LRC", "RFC", "SVC", "GBC"]`.
* No código base original, a chave `"SVC"` instanciava `LinearSVC()`, embora `SVC` estivesse importado sem uso. Em [`_get_models_and_grids`](./pds-model/src/pds/pipeline.py), a chave `"SVC"` passou a instanciar explicitamente `sklearn.svm.SVC(probability=True, random_state=RANDOM_STATE)`, avaliando os kernels `["linear", "rbf"]` ao longo da grade de regularização `C`.

### 2.2. Cálculo Correto de `ROC-AUC` (Scores Contínuos) e `Brier Score` (Probabilidades)
* O código original passava os rótulos binários discretos (`y_pred` $\in \{0, 1\}$) diretamente para `metrics.roc_auc_score(y_test, y_pred)` e `metrics.brier_score_loss(y_test, y_pred)`, o que reduzia o ROC-AUC à acurácia balanceada de um único limiar e o Brier Score à taxa de erro de classificação (`1 - accuracy`).
* A função [`_predict_with_scores`](./pds-model/src/pds/pipeline.py) extrai as probabilidades contínuas da classe positiva (`predict_proba(X_test)[:, 1]` ou `decision_function`), repassando `y_score` e `y_proba` para [`extract_model_metrics_scores`](./pds-model/src/pds/pipeline.py). Assim, `roc_auc_score` avalia o ranqueamento contínuo completo do classificador e `brier_score_loss` mede genuinamente a calibração probabilística, garantindo que [`select_best_model`](./pds-model/src/pds/pipeline.py) escolha o modelo campeão pela métrica correta.

### 2.3. Eliminação de Vazamento de Dados (*Data Leakage*) no `SMOTE` e Escalonamento
* No código original, `SMOTE().fit_resample(X_train, y_train)` era aplicado sobre todo o `X_train` **antes** do `GridSearchCV(cv=5)`. Isso fazia com que amostras sintéticas interpoladas entre diferentes regiões de `X_train` caíssem simultaneamente nos folds internos de treino e validação do `GridSearchCV`, enviesando a escolha de hiperparâmetros. Além disso, a ausência de escalonamento prévio distorcia a distância euclidiana dos $k$-vizinhos do `SMOTE` (dominada por `LOC` na casa dos milhares) e causava colapso de convergência em modelos lineares (`LRC` e `SVC`).
* **Solução implementada:** Em [`_prepare_leak_free_inner_cv`](./pds-model/src/pds/pipeline.py) e [`_build_estimator_pipeline`](./pds-model/src/pds/pipeline.py), tanto o `StandardScaler` quanto o `SMOTE(random_state=RANDOM_STATE)` são ajustados **exclusivamente dentro da partição de treino de cada fold interno** (sem jamais gerar amostras sintéticas na partição de validação interna). Ao final da seleção de hiperparâmetros, o modelo vencedor é encapsulado e ajustado como um `imblearn.pipeline.Pipeline` (`StandardScaler` $\rightarrow$ `SMOTE` $\rightarrow$ `Classificador`), cuja equivalência numérica exata (`diff = 0.00e+00`) com `GridSearchCV(ImbPipeline(...))` é verificada pela suíte de testes unitários em [`test_pipeline.py`](./pds-model/tests/test_pipeline.py).

### 2.4. Reprodutibilidade Determinística (`RANDOM_STATE = 42`) e Paralelização (`n_jobs=-1`)
* Definida a constante global `RANDOM_STATE = 42` em [`pipeline.py`](./pds-model/src/pds/pipeline.py), fixando as sementes pseudoaleatórias em `StratifiedKFold` (externo de 10 folds e interno de 5 folds com `shuffle=True`), `train_test_split(..., stratify=y)`, `SMOTE(random_state=42)` e nos quatro classificadores (`SVC`, `LRC`, `RFC`, `GBC`).
* Configurado `n_jobs=-1` no `GridSearchCV` e pré-computação dos 5 folds internos por fold externo, viabilizando a execução dos **2.116.800 ajustes** do `GBC` ($42.336 \times 5 \times 10$) nos 16 threads do processador Intel Core i5-13450HX.

---

## 3. Resultados Experimentais: Benchmark 10-Fold Stratified CV

Ambos os datasets foram submetidos exatamente ao mesmo protocolo experimental em [`experiment_runner.py`](./pds-model/experiment_runner.py): 10 folds estratificados (`StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`), com `StandardScaler` e `SMOTE(random_state=42)` isolados por fold interno de 5 partições e seleção de hiperparâmetros por `roc_auc`.

### Tabela 1 — Benchmark Comparativo dos 4 Classificadores (10 Folds — Média $\pm$ Desvio Padrão)

| Dataset | Modelo | ROC-AUC (Contínuo) | Brier Score (Prob.) | F1-Score | Acurácia | Precisão | Recall | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ATP05 (`v3.0.6`)**<br>*(230 módulos reais)* | **GBC** 🏆 | **0,8352 $\pm$ 0,0544** | 0,1987 $\pm$ 0,0266 | 0,6581 $\pm$ 0,0726 | 0,6957 | 0,6765 | **0,6636** | 0,3972 |
| | **SVC** | 0,8339 $\pm$ 0,0756 | **0,1677 $\pm$ 0,0301** | 0,6663 $\pm$ 0,1454 | **0,7609** | **0,8685** | 0,5582 | **0,5357** |
| | **RFC** | 0,8234 $\pm$ 0,0531 | 0,1687 $\pm$ 0,0265 | **0,6864 $\pm$ 0,0751** | 0,7348 | 0,7446 | 0,6536 | 0,4713 |
| | **LRC** | 0,8225 $\pm$ 0,0810 | 0,2223 $\pm$ 0,0222 | 0,6601 $\pm$ 0,1542 | 0,7565 | 0,8477 | 0,5573 | 0,5203 |
| **BASELINE (`v2.3.1`)**<br>*(1.402 arquivos c/ 73% testes)* | **RFC** 🏆 | **0,8083 $\pm$ 0,0337** | **0,1795 $\pm$ 0,0151** | **0,7265 $\pm$ 0,0407** | **0,7268** | 0,7591 | **0,6977** | **0,4569** |
| | **GBC** | 0,7999 $\pm$ 0,0316 | 0,1843 $\pm$ 0,0126 | 0,7141 $\pm$ 0,0333 | 0,7183 | 0,7615 | 0,6744 | 0,4427 |
| | **SVC** | 0,7934 $\pm$ 0,0354 | 0,1885 $\pm$ 0,0124 | 0,6645 $\pm$ 0,0338 | 0,7054 | **0,8186** | 0,5608 | 0,4426 |
| | **LRC** | 0,7864 $\pm$ 0,0394 | 0,1889 $\pm$ 0,0134 | 0,6762 $\pm$ 0,0293 | 0,7104 | 0,8129 | 0,5800 | 0,4478 |

---

## 4. Avaliação dos Modelos Campeões no Conjunto de Teste Hold-out (Split 75/25)

Conforme a lógica de `make_model(...)`, o classificador de maior média de ROC-AUC em validação cruzada é ajustado na partição estratificada de treino (75% dos dados) e avaliado na partição final de teste (25% não vistos).

### Tabela 2 — Comparação Direta dos Modelos Campeões no Teste Hold-Out

| Métrica de Desempenho | Campeão ATP05 (`v3.0.6`)<br>**GBC (Gradient Boosting)** | Campeão Baseline (`v2.3.1`)<br>**RFC (Random Forest)** | Diferença Absoluta ($\Delta$) | Impacto Prático |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC Score (Contínuo)** | **0,8888** | 0,8128 | **+0,0760 (+7,60 p.p.)** | Superior capacidade discriminativa global |
| **Brier Score Loss (Prob.)** | **0,1707** | 0,1802 | **-0,0095 (Menor erro)** | Probabilidades mais bem calibradas |
| **Acurácia Global** | **0,7931 (79,31%)** | 0,7464 (74,64%) | **+0,0467 (+4,67 p.p.)** | Menor taxa de erros absolutos |
| **F1-Score (Classe Positiva)** | **0,7692** | 0,7390 | **+0,0302 (+3,02 p.p.)** | Melhor equilíbrio entre precisão e recall |
| **Precisão (`BUG=1`)** | 0,7692 (76,92%) | **0,7975 (79,75%)** | -0,0283 (-2,83 p.p.) | Alta confiabilidade nos alertas emitidos |
| **Recall (`BUG=1`)** | **0,7692 (76,92%)** | 0,6885 (68,85%) | **+0,0807 (+8,07 p.p.)** | Expressivo ganho na detecção de módulos defeituosos |
| **Matthews Corr. Coef. (MCC)** | **0,5817** | 0,5001 | **+0,0816 (+16,3% relativo)** | Correlação preditiva substancialmente superior |

### 4.1. Matrizes de Confusão e Hiperparâmetros dos Campeões no Teste

#### Campeão ATP05 (`v3.0.6` — GBC)
* **Hiperparâmetros ótimos selecionados:** `learning_rate=0.075`, `max_depth=5`, `max_features='log2'`, `min_samples_leaf=0.1`, `min_samples_split=0.3182`, `subsample=0.85`, `n_estimators=10`, `criterion='friedman_mse'`, `loss='log_loss'`.
```
Matriz de Confusão:
                Previsto: SEM BUG (0)    Previsto: COM BUG (1)
Real: SEM BUG           26 (VN)                  6 (FP)
Real: COM BUG            6 (FN)                 20 (VP)

Total de amostras de teste: 58 arquivos (32 sem bug, 26 com bug)
Acurácia: 79,31% | Precisão: 76,92% | Recall: 76,92% | F1: 0,7692
```

#### Campeão Baseline (`v2.3.1` — RFC)
* **Hiperparâmetros ótimos selecionados:** `n_estimators=100`, `max_depth=6`, `criterion='gini'`, `max_features='sqrt'`.
```
Matriz de Confusão:
                Previsto: SEM BUG (0)    Previsto: COM BUG (1)
Real: SEM BUG          136 (VN)                 32 (FP)
Real: COM BUG           57 (FN)                126 (VP)

Total de amostras de teste: 351 arquivos (168 sem bug, 183 com bug)
Acurácia: 74,64% | Precisão: 79,75% | Recall: 68,85% | F1: 0,7390
```

---

## 5. Análise Crítica e Diagnóstico Metodológico

### 5.1. O Impacto da Limpeza de Dados da ATP05 nos Modelos
* **Superioridade Sistemática em Todos os 4 Classificadores:** Todos os algoritmos avaliados (`GBC`, `SVC`, `RFC` e `LRC`) atingiram ROC-AUC médio em CV superior no dataset tratado `v3.0.6` (**0,8225 a 0,8352**) em comparação ao dataset baseline `v2.3.1` (**0,7864 a 0,8083**).
* **Eliminação de Ruído de Testes Unitários:** No dataset original, 73% das amostras eram arquivos de teste cujas métricas estruturais criavam fronteiras de decisão ruidosas. Ao focar nos 230 módulos ativos de produção da release `v3.0.6`, o sinal estrutural entre volume/complexidade (`LOC`, `CYC`, `NER`) e incidência de bugs emergiu com nitidez, elevando o ROC-AUC em hold-out de **0,8128 para 0,8888 (+7,60 p.p.)** e o Recall de **68,85% para 76,92% (+8,07 p.p.)**.

### 5.2. O Papel dos Novos Classificadores (`GBC` e `SVC`) e do Pré-processamento
1. **Gradient Boosting Classifier (`GBC`):** Sagrou-se o **modelo campeão geral** sobre o dataset da ATP05 (**ROC-AUC 0,8352** em CV e **0,8888** no teste), superando os dois modelos que existiam originalmente no setup (`RFC` com 0,8234 e `LRC` com 0,8225).
2. **Support Vector Classifier (`sklearn.svm.SVC`):** Alcançou o **2º lugar geral** na ATP05 (**ROC-AUC 0,8339** em CV, a apenas 0,0013 do `GBC`), superando tanto `RFC` quanto `LRC` e obtendo a menor perda de calibração probabilística em CV (**Brier = 0,1677**) e o maior MCC (**0,5357**).
3. **Efeito do `StandardScaler` acoplado ao `ImbPipeline`:** Com a padronização de escala aplicada dentro de cada fold interno antes do `SMOTE`, a distância euclidiana dos vizinhos sintéticos deixou de ser dominada exclusivamente por `LOC`, e a Regressão Logística (`LRC`) deixou de sofrer colapso por penalização $L_1$, atingindo **ROC-AUC 0,8225**.

---

## 6. Conclusões e Artefatos Entregáveis

1. **Confirmação da Hipótese da Prática:** A adição de `SVC` e `GBC` ao benchmark levou diretamente a modelos de melhor performance que os originais (`LRC` e `RFC`) sobre o dataset de alta qualidade da ATP05, com o **`GBC`** consagrando-se campeão (**ROC-AUC 0,8352 em CV / 0,8888 em Hold-Out**).
2. **Artefatos do Repositório e Entrega SIGAA:**
   * Código final para submissão no SIGAA: [`pipeline.py`](./pds-model/src/pds/pipeline.py).
   * Suíte de testes automatizados de validação metodológica: [`test_pipeline.py`](./pds-model/tests/test_pipeline.py).
   * Script de execução comparativa completa: [`experiment_runner.py`](./pds-model/experiment_runner.py).
   * Resultados completos e logs de ambos os datasets: [`results_comparison.json`](./pds-model/results_comparison.json) e [`pipeline.log`](./pds-model/pipeline.log).
   * Modelos serializados (`.pkl`): `model_GBC_ROC_AUC_SMOTE.pkl` (campeão geral), `model_ATP05_v3_0_6_GBC_ROC_AUC_SMOTE.pkl` e `model_BASELINE_v2_3_1_RFC_ROC_AUC_SMOTE.pkl`.

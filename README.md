# Atividade Prática 06 — Melhorias na Pipeline de Modelos

> **Disciplina:** Engenharia de Sistemas Inteligentes (CK0444 – 2026.2)  
> **Instituição:** Departamento de Computação (DC) — Universidade Federal do Ceará (UFC)

Objetivos da prática:

- Aprofundar o conhecimento sobre pipeline de modelos de Machine Learning para Predição de Defeitos de Software (PDS).
- Remodelagem da pipeline de modelos adicionando mais classificadores (`SVC` e `GBC`) e garantindo rigor metodológico na validação cruzada.

## 1. Pré-requisitos

### 1.1. Poetry Instalado

Verifique se o Poetry está instalado no seu terminal de comandos (`shell` ou `console`):

```bash
poetry --version
```

Caso não esteja instalado, utilize o comando abaixo para realizar a instalação:

```bash
pip install poetry
```

### 1.2. Estrutura do Projeto

O subprojeto `pds-model` encontra-se no diretório [`./pds-model`](./pds-model), acompanhado do relatório técnico detalhado em [`DECISOES_PROJETO.md`](./DECISOES_PROJETO.md).

### 1.3. Visual Studio Code (VS Code)

Recomendamos o uso da IDE VS Code para execução desta prática.

## 2. Roteiro da Prática

### 2.1. Modificando o Set-up de Experimentação

Na função `start(...)`, a chamada a `do_benchmark(...)` foi expandida de `["LRC", "RFC"]` para `["LRC", "RFC", "SVC", "GBC"]`, incorporando `sklearn.svm.SVC` e `GradientBoostingClassifier` para avaliar se essa modificação leva a um modelo final de melhor performance.

**Nota:** A performance dos modelos é comparada entre o dataset de referência (`v2_3_1_trf_dataset.csv`) e o dataset de melhor qualidade gerado na `ATP05` (`v3_0_6_trf_dataset.csv`).

### 2.2. Entrega da Tarefa

A versão final de [`pipeline.py`](./pds-model/src/pds/pipeline.py) de `pds-model` conduz ao modelo de melhor performance e está pronta para submissão.

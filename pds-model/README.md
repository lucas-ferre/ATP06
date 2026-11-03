# Pipeline de Modelos para o Sistema de PDS

> **Curso:** Bacharelado em Ciência de Dados  
> **Disciplina:** Engenharia de Sistemas Inteligentes (CK0444 – 2026.2)  
> **Instituição:** Departamento de Computação (DC) — Universidade Federal do Ceará (UFC)

A pipeline de modelos ([pipeline.py](./src/pds/pipeline.py)) foi escrita para escolher o modelo de ML de melhor desempenho para a tarefa de Predição de Defeitos de Software (PDS) por meio de um experimento de benchmark em validação cruzada estratificada. A pipeline usa como entrada o dataset rotulado gerado pela pipeline de dados e produz como saída o modelo de ML campeão serializado (`.pkl`). O projeto `pds-model` segue a seguinte estrutura:

```bash
pds-model
├── src
│   └── pds
│       ├── __init__.py
│       └── pipeline.py
├── tests
│   ├── __init__.py
│   └── test_pipeline.py
├── pyproject.toml
└── README.md
```

Instruções para execução da pipeline de modelos:

1. Instale o Poetry:

```bash
pip install poetry
```

2. Instale as dependências necessárias para rodar a pipeline de modelos:

```bash
poetry install
```

3. Execute a pipeline de modelos:

```bash
poetry run modelpip "<caminho do dataset.csv>"
```

# Projeto 3 — Detecção de Máscaras Faciais (YOLO)

## 💻 O Desafio Técnico

Desenvolva um modelo de **detecção de objetos** capaz de identificar, em uma
imagem com rostos, se cada pessoa está **usando máscara corretamente**, **sem
máscara**, ou **usando a máscara de forma incorreta** — localizando cada rosto
com uma bounding box.

Diferente dos Projetos 1 e 2 (onde você constrói uma CNN do zero), aqui o
objetivo é **adaptar e otimizar um framework de detecção real para Edge AI** —
uma competência bastante prática no dia a dia de Visão Computacional Embarcada,
já que a imensa maioria das aplicações de detecção em produção parte de um
modelo pré-treinado, não de uma arquitetura construída do zero.

> ⚠️ **Exceção importante:** ao contrário dos Projetos 1 e 2, aqui o uso de
> **pesos pré-treinados é permitido e esperado** (fine-tuning). Isso é
> intencional — este projeto avalia uma competência diferente: adaptar,
> treinar e exportar um framework de detecção real para o seu dataset.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**fine-tuning → validação → exportação → otimização para edge**

## 🎯 Conjunto de Dados

Este projeto já vem com um dataset **pronto para uso**, na pasta [`dataset/`](dataset/):
o **Face Mask Detection Dataset** ([Kaggle, andrewmvd](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection),
licença **CC0 1.0** — domínio público), já convertido do formato original (Pascal VOC)
para o formato esperado pelo Ultralytics YOLO.

- **853 imagens** de rostos, com bounding boxes anotadas
- **3 classes:** `with_mask`, `without_mask`, `mask_weared_incorrect`
- Já dividido em treino (~80%) e validação (~20%)
- ⚠️ O dataset é **desbalanceado** — a classe `mask_weared_incorrect` tem
  significativamente menos exemplos que as outras duas. Isso é uma
  característica real de datasets de detecção e não é um bug — comente esse
  ponto no seu relatório se perceber o modelo com dificuldade nessa classe.

Você **não precisa** baixar nada do Kaggle nem escrever código de conversão de
anotações — isso já está pronto em `dataset/`. Seu trabalho começa direto no
fine-tuning do modelo.

## ✅ Requisitos Obrigatórios

### Etapa 1 — Fine-tuning do Modelo (`train_model.py`)

Implemente, usando a biblioteca **Ultralytics** (YOLO):

- Carregamento do modelo pré-treinado **YOLO11n** (`YOLO("yolo11n.pt")`) —
  esta é a única exceção à regra de "sem modelos pré-treinados" do processo
  seletivo, válida especificamente para este projeto
- Fine-tuning no dataset fornecido (`dataset/data.yaml`), em **CPU**, com um
  número de épocas modesto (ex: 15-30 — YOLO converge relativamente rápido
  em fine-tuning, mesmo em CPU)
- Ao final do treino, copie os pesos resultantes (`runs/detect/train/weights/best.pt`)
  para a raiz desta pasta, com o nome **`model.pt`**

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.pt` treinado
- Exportação para **TensorFlow Lite** via `model.export(format="tflite")`
  (a Ultralytics gera automaticamente um arquivo `model.tflite` na mesma pasta)

> 💡 Na primeira execução, a Ultralytics pode instalar automaticamente
> dependências extras necessárias para a exportação (isso é esperado e pode
> levar alguns minutos).

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.pt`) usando `YOLO("model.tflite", task="detect")`
- Execução de inferência em pelo menos **5 imagens** de `dataset/images/val/`,
  **uma de cada vez** — o `model.tflite` exportado aceita apenas 1 imagem por
  chamada (batch=1), que é aliás o cenário real de uso em edge
- Exibição no terminal, para cada imagem, do número de detecções encontradas

> 💡 O Ultralytics salva automaticamente as imagens anotadas com as caixas
> preditas em `runs/detect/...` (pasta já ignorada pelo `.gitignore` — não
> precisa, nem deve, ser commitada). Abra essas imagens localmente pra conferir
> visualmente as predições antes de escrever o relatório.
>
> 💡 Essa etapa existe porque uma métrica agregada (mAP) pode esconder
> problemas que só aparecem olhando exemplos individuais — especialmente dado
> o desbalanceamento de classes deste dataset.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos nem a estrutura de `dataset/`.

```
projetos/3-deteccao-mascaras/
├── train_model.py         # ✏️ Fine-tuning do modelo
├── optimize_model.py      # ✏️ Exportação e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.pt               # 🤖 Gerado por você — deve ser commitado
├── model.tflite            # ⚡ Gerado por você — deve ser commitado
├── README.md               # 📝 Este arquivo (também usado como relatório)
└── dataset/                # 📦 Dataset já pronto (não modificar)
    ├── data.yaml
    ├── images/{train,val}/
    └── labels/{train,val}/
```

## ⚠️ Restrições e Considerações de Engenharia

- Modelo base: **YOLO11n** (variante *nano*, indicada para CPU/edge) — não use
  variantes maiores (s/m/l/x)
- Treinamento apenas em CPU
- Fine-tuning é permitido e esperado (única exceção às regras gerais do processo seletivo)
- **Não é esperada detecção perfeita**, especialmente na classe minoritária
  (`mask_weared_incorrect`) — o objetivo é demonstrar que o pipeline completo
  (fine-tuning → validação → exportação) funciona corretamente
- O tempo de treinamento e exportação deste projeto tende a ser **maior** que
  o dos Projetos 1 e 2 — reserve tempo extra para rodar localmente antes de enviar

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração de `model.pt` e `model.tflite`
- **Qualidade do modelo** — mAP50 no conjunto de validação acima do mínimo esperado
- **Edge AI** — exportação correta para `.tflite`
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Júlia Novais Pereira

### 1️⃣ Resumo da Abordagem

- **Modelo Base:** YOLO11n (`yolo11n.pt`), adaptado via fine-tuning em CPU.
- **Hiperparâmetros:** 
  - Tamanho de Imagem (`imgsz`): 640x640
  - Épocas: 20 épocas
  - Batch Size: 16
- **Tratamento de Desbalanceamento:** Não foram aplicados pesos customizados por classe no código para manter o padrão puro do pipeline YOLO, permitindo observar o comportamento natural da arquitetura diante da classe minoritária (`mask_weared_incorrect`).

### 2️⃣ Bibliotecas Utilizadas

### 2️⃣ Bibliotecas Utilizadas

- **`ultralytics` (v8.4.106):** Framework principal utilizado para fine-tuning, validação, inferência e exportação do modelo YOLO.
- **`os` & `shutil`:** Bibliotecas nativas do Python utilizadas no fluxo de scripts para verificação de diretórios e manipulação/cópia de arquivos de pesos (`model.pt`).
- **`torch` & `torchvision`:** Dependências de backend do PyTorch gerenciadas pelo Ultralytics para processamento gráfico e algoritmos de NMS (Non-Maximum Suppression).
- **`ai-edge-litert` & `litert-torch`:** Suporte de backend carregado pelo Ultralytics no ambiente Linux para realizar a exportação e inferência no formato otimizado Google LiteRT/TFLite.

### 3️⃣ Técnica de Otimização do Modelo

A conversão do modelo treinado para o formato de borda foi realizada através da instrução `model.export(format="tflite", imgsz=640)`. 

O ecossistema Ultralytics realizou a conversão do grafo PyTorch (`.pt`) para o formato unificado **Google LiteRT** (`.tflite`). Como o suporte a essa exportação direta possui limitações no ecossistema Windows, o processo foi executado dentro de um **Dev Container (Docker + WSL2 em ambiente Linux Ubuntu)**, garantindo a geração correta dos FlatBuffers e a otimização com o delegado XNNPACK para inferência ultra-rápida em CPU..

### 4️⃣ Resultados Obtidos

### Métricas de Validação (mAP):
- **Geral (all):** 
  - **Precision:** 0.837 | **Recall:** 0.661 | **mAP50:** 0.749 | **mAP50-95:** 0.517
- **Por Classe:**
  - `with_mask`: Precision: 0.940 | Recall: 0.931 | **mAP50: 0.969** | **mAP50-95: 0.674**
  - `without_mask`: Precision: 0.889 | Recall: 0.630 | **mAP50: 0.783** | **mAP50-95: 0.496**
  - `mask_weared_incorrect`: Precision: 0.682 | Recall: 0.421 | **mAP50: 0.495** | **mAP50-95: 0.381**

### Tamanho dos Arquivos:
- `model.pt`: **5.2 MB**
- `model.tflite`: **10.1 MB**

### 5️⃣ Comentários Adicionais (Opcional)

- **Desafios Enfrentados:** A exportação nativa para TFLite pelo ecossistema do Ultralytics possui suporte restrito ao Windows. Para contornar essa limitação e simular o ambiente real de Edge/Linux, o desenvolvimento foi migrado para um **Dev Container (Docker + WSL2)**, viabilizando o uso do Google LiteRT.
- **Comportamento da Classe Minoritária:** O modelo identificou corretamente instâncias de `mask_weared_incorrect` em casos evidentes (máscara no queixo), mas apresentou menor nível de confiança (*confidence score*) e confusão ocasional com `with_mask` em oclusões parciais.

### 6️⃣ Exemplo de Inferência

```text
============================================================
Projeto 3 — Inferência com model.tflite (Edge AI)
============================================================

Rodando inferência em 5 amostras usando model.tflite:

Imagem                               Detecções  Detalhes
----------------------------------------------------------------------
Loading /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/model.tflite for LiteRT inference...
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
maksssksksss105.jpg                          9  [9x with_mask]
maksssksksss107.jpg                          1  [1x with_mask]
maksssksksss11.jpg                          23  [21x with_mask, 2x mask_weared_incorrect]
maksssksksss113.jpg                          4  [3x with_mask, 1x without_mask]
maksssksksss12.jpg                          13  [11x with_mask, 2x without_mask]
----------------------------------------------------------------------
TOTAL                                       50

✅ Imagens anotadas salvas em: runs/detect/inferencia_exemplos/predicoes/
   (Abra essa pasta para verificar visualmente as bounding boxes preditas)
```
Bounding Boxes: Muito bem localizadas e ajustadas aos rostos, mesmo em multidões e em rostos menores ao fundo.

Classes Principais: Excelente separação entre with_mask e without_mask, sem confusões relevantes.

Classe Minoritária (mask_weared_incorrect): O modelo detectou casos reais de uso incorreto, mas com confiança mais baixa e alguns falsos negativos, refletindo o desbalanceamento do dataset.
---

## 📄 Créditos do Dataset

Face Mask Detection Dataset — [Kaggle: andrewmvd/face-mask-detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection), licença CC0 1.0 (domínio público).

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Projeto 3 — Otimização do Modelo (Exportação para Edge)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.pt"
#   2. Exportar para TensorFlow Lite via model.export(format="tflite")
#      (a Ultralytics gera automaticamente "model.tflite" na mesma pasta)
# ---------------------------------------------------------------------------

# insira seu código aqui

# Dica de estrutura (não é obrigatório seguir exatamente assim):
#
# model = YOLO("model.pt")
# model.export(format="tflite", imgsz=...)


from ultralytics import YOLO

# 1. Carrega o modelo treinado em "model.pt"
model = YOLO("model.pt")

# 2. Exporta para TensorFlow Lite
# A Ultralytics gera o arquivo automaticamente
model.export(format="tflite", imgsz=640)

print("✅ Modelo exportado com sucesso!")
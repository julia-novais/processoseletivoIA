import os       # <--- Para checar arquivos e caminhos de pastas
import shutil   # <--- Para fazer a cópia do arquivo
from ultralytics import YOLO

# 1. Carrega o modelo pré-treinado
model = YOLO("yolo11n.pt")

# 2. Roda o treinamento
results = model.train(
    data="dataset/data.yaml",
    epochs=20,
    device="cpu",
    imgsz=640,
    name="mask_detection"
)

# 3. Monta o caminho do arquivo gerado
# results.save_dir devolve a pasta do treino (ex: runs/detect/mask_detection)
caminho_origem = os.path.join(results.save_dir, "weights", "best.pt")
caminho_destino = "model.pt"

# 4. IMPLEMENTAÇÃO DO 'os': Checa se o arquivo de fato existe antes de copiar
if os.path.exists(caminho_origem):
    shutil.copy(caminho_origem, caminho_destino)
    print(f"✅ Treinamento concluído com sucesso!")
    print(f"📦 Arquivo copiado de '{caminho_origem}' para '{caminho_destino}'.")
else:
    print(f"⚠️ Erro: O arquivo '{caminho_origem}' não foi encontrado!")

# shutil.copy(results.save_dir / "weights" / "best.pt", "model.pt")

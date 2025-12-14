import os
import io

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from starlette.responses import Response

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models import create_model
from basicsr.utils.options import parse


app = FastAPI()

# -----------------------
# 모델 공통 설정
# -----------------------
OPT_PATH = "Options/RetinexFormer_LOL_v2_real.yml"
WEIGHTS_PATH = "pretrained_weights/LOL_v2_real.pth"
FACTOR = 4  # 패딩에 사용

# 사용할 GPU 설정 (예: "0" 또는 "0,1")
GPU_LIST = os.getenv("RETINEX_GPUS", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_LIST
print("export CUDA_VISIBLE_DEVICES=" + GPU_LIST)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[RetinexService] Using device: {DEVICE}")


def self_ensemble(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """test_from_dataset.py 상단에 있는 self_ensemble 그대로 복사"""
    def forward_transformed(x, hflip, vflip, rotate, model):
        if hflip:
            x = torch.flip(x, (-2,))
        if vflip:
            x = torch.flip(x, (-1,))
        if rotate:
            x = torch.rot90(x, dims=(-2, -1))
        x = model(x)
        if rotate:
            x = torch.rot90(x, dims=(-2, -1), k=3)
        if vflip:
            x = torch.flip(x, (-1,))
        if hflip:
            x = torch.flip(x, (-2,))
        return x

    t = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                t.append(forward_transformed(x, hflip, vflip, rot, model))
    t = torch.stack(t)
    return torch.mean(t, dim=0)


# self_ensemble 사용 여부 (기본 False, 환경변수로 켤 수 있음)
USE_SELF_ENSEMBLE = os.getenv("RETINEX_SELF_ENSEMBLE", "0") == "1"

# -----------------------
# 모델 로딩
# -----------------------
print("[RetinexService] Loading model config...")

opt = parse(OPT_PATH, is_train=False)
opt["dist"] = False

print("[RetinexService] Creating model...")
_model_wrapper = create_model(opt)
model_restoration: nn.Module = _model_wrapper.net_g

print("[RetinexService] Loading weights from", WEIGHTS_PATH)
checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")
params = checkpoint.get("params", checkpoint)

try:
    model_restoration.load_state_dict(params)
except Exception:
    # 일부 체크포인트는 key 앞에 'module.' 이 없어서 예외 처리
    new_checkpoint = {}
    for k, v in params.items():
        new_checkpoint["module." + k] = v
    model_restoration.load_state_dict(new_checkpoint)

model_restoration.to(DEVICE)
if DEVICE == "cuda":
    model_restoration = nn.DataParallel(model_restoration)

model_restoration.eval()
print("[RetinexService] Model ready.")


# -----------------------
# 단일 이미지 추론
# -----------------------
def run_retinexformer_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """
    입력: BGR OpenCV 이미지 (uint8)
    출력: BGR OpenCV 이미지 (uint8)
    test_from_dataset.py 의 else 브랜치 로직을 단일 이미지용으로 옮긴 것.
    BGR -> RGB -> model -> RGB -> BGR 변환을 포함한다.
    """
    if img_bgr is None:
        raise ValueError("run_retinexformer_bgr: input image is None")

    # 1) BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2) 0~255 uint8 -> 0~1 float32
    img = np.float32(img_rgb) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        # 패딩 (4의 배수)
        b, c, h, w = img.shape
        H = ((h + FACTOR) // FACTOR) * FACTOR
        W = ((w + FACTOR) // FACTOR) * FACTOR
        padh = H - h if h % FACTOR != 0 else 0
        padw = W - w if w % FACTOR != 0 else 0
        input_ = F.pad(img, (0, padw, 0, padh), "reflect")

        # 원 코드에서 h,w<3000 이면 그대로, 아니면 split
        if h < 3000 and w < 3000:
            if USE_SELF_ENSEMBLE:
                restored = self_ensemble(input_, model_restoration)
            else:
                restored = model_restoration(input_)
        else:
            # 큰 경우는 열을 반으로 나눠서 처리
            input_1 = input_[:, :, :, 1::2]
            input_2 = input_[:, :, :, 0::2]
            if USE_SELF_ENSEMBLE:
                restored_1 = self_ensemble(input_1, model_restoration)
                restored_2 = self_ensemble(input_2, model_restoration)
            else:
                restored_1 = model_restoration(input_1)
                restored_2 = model_restoration(input_2)
            restored = torch.zeros_like(input_)
            restored[:, :, :, 1::2] = restored_1
            restored[:, :, :, 0::2] = restored_2

        # 패딩 제거
        restored = restored[:, :, :h, :w]

        restored = torch.clamp(restored, 0, 1).detach().cpu()
        restored = restored.permute(0, 2, 3, 1).squeeze(0).numpy()

    # 3) 0~1 float -> 0~255 uint8, RGB 기준
    out_rgb = (restored * 255.0).round().astype(np.uint8)

    # 4) RGB -> BGR 로 되돌려서 OpenCV 스타일로 반환
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    return out_bgr


# -----------------------
# FastAPI 엔드포인트
# -----------------------
@app.post("/enhance")
async def enhance(image: UploadFile = File(...)):
    """
    worker/pipeline.py 의 _call_image_model 이 기대하는 프로토콜:
      - multipart/form-data 로 "image" 필드에 파일이 담겨 옴
      - 응답은 처리된 이미지를 PNG 바이너리로 반환
    """
    data = await image.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return Response(status_code=400, content=b"invalid image")

    out = run_retinexformer_bgr(img)

    ok, buf = cv2.imencode(".png", out)
    if not ok:
        return Response(status_code=500, content=b"encode error")

    return Response(content=buf.tobytes(), media_type="image/png")

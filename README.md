# RetinexFormer (model service)

저조도 개선(RetinexFormer) 모델을 HTTP 서비스로 띄우는 폴더입니다.

- 기본 포트: `9000`
- 엔드포인트: `POST /enhance`
- backend는 이 서비스를 `RETINEX_URL`로 호출합니다.

## 가중치/옵션 파일 위치

서비스 코드는 아래 경로를 **상대경로로 고정**해서 읽습니다 (저희는 LOL_v2_real 가중치를 사용했지만 다른걸 사용해도 무방합니다).

- 옵션(yml): `Options/RetinexFormer_LOL_v2_real.yml`
- 가중치(pth): `pretrained_weights/LOL_v2_real.pth`

레포에는 폴더만 포함되어 있고, 실제 가중치는 직접 넣어야 합니다.

## Conda 환경

GPU(CUDA 11.8) 환경:
```bash
conda env create -f ../envs/fridai-retinexformer-cu118.yml
conda activate fridai-retinexformer
```

CPU 환경:
```bash
conda env create -f ../envs/fridai-retinexformer-cpu.yml
conda activate fridai-retinexformer
```

## 실행

```bash
uvicorn retinex_service:app --host 0.0.0.0 --port 9000
```

## 환경변수

- `RETINEX_GPUS` : 멀티 GPU 사용 설정(있으면 그대로 전달)
- `RETINEX_SELF_ENSEMBLE` : self-ensemble 사용 여부(기본: 꺼짐)

# Product Parsing Pipeline (Updated Prompts)

이 프로젝트는 상품 페이지의 텍스트와 이미지를 크롤링하여,
상품 정보, 이미지 분류, 이미지 임베딩, 표/텍스트 추출 등 다양한 정보를 구조화된 JSON으로 반환하는 파이프라인입니다.

## 주요 기능

- **동적 웹페이지 크롤링**: Selenium을 사용하여 자바스크립트 렌더링이 필요한 페이지도 완벽하게 크롤링
- **텍스트/이미지 추출**: 페이지 내 텍스트와 모든 이미지를 다운로드 및 저장
- **이미지 전처리**:
  - 너무 작은 이미지는 자동 필터링 (width < 100px 또는 height < 100px)
  - 세로로 긴 이미지는 자동 분할(겹침 포함)하여 처리
- **이미지 분류**: 각 이미지를 `product photo`, `table with text`, `text-only image`, `unrelated graphic` 등으로 분류
- **임베딩 및 OCR/표 추출**:
  - `product photo`로 분류된 모든 이미지에 대해 CLIP 임베딩(옵션)
  - 표/텍스트 이미지는 Table Detection, OCR 등으로 정보 추출
- **구조화된 결과**: 모든 결과를 JSON 파일로 저장 (DB 연동 가능)

---

## 설치 및 환경

### 1. 필수 패키지
- Python 3.8+
- [Selenium](https://pypi.org/project/selenium/)
- [Pillow](https://pypi.org/project/Pillow/)
- [BeautifulSoup4](https://pypi.org/project/beautifulsoup4/)
- [requests](https://pypi.org/project/requests/)
- torch, 기타 모델 관련 패키지

```bash
pip install selenium pillow beautifulsoup4 requests torch
```

### 2. Chrome/Chromium 및 chromedriver
- 서버에 Chrome(또는 Chromium)과 chromedriver가 설치되어 있어야 합니다.
- chromedriver가 PATH에 없으면, 환경변수 또는 코드에서 직접 경로 지정 필요

---

## 실행 방법

### 1. 단일/복수 URL 처리

```bash
python -m parser_service_updated_prompts.main <URL1> <URL2> ... [OPTIONS]
```

#### 예시
```bash
python -m parser_service_updated_prompts.main "https://store.kakao.com/sleepnsleep/products/429138587" --embed_product_photos
```

### 2. 주요 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--results_dir` | 결과 JSON 저장 디렉토리 | `./parsing_results_updated` |
| `--workers` | 동시 처리 워커 수 | 2 |
| `--debug` | 디버그 로그 출력 | - |
| `--embed_product_photos` | product photo 이미지의 CLIP 임베딩 저장 여부 | 미지정시 저장 안함 |

---

## 결과 구조 예시

```json
{
  "meta": {
    "uuid": "xxxx-xxxx-xxxx",
    "url": "https://...",
    "timestamp": "2024-06-07T12:34:56"
  },
  "product_text_info": {
    "title": "...",
    "description": "..."
  },
  "product_images_info": {
    "images_embedding": [
      {
        "img_url": "https://original.image.url/1.jpg",
        "clip_embedding": [0.123, 0.456, ...]
      },
      ...
    ],
    "images_text": [
      "텍스트 이미지에서 추출된 텍스트1"
    ],
    "tables_text": [
      "표 이미지에서 추출된 자연어 설명1"
    ]
  }
}
```
- `images_embedding`은 `--embed_product_photos` 옵션을 줄 때만 생성됩니다.

---

## 커스텀/확장 포인트

- **이미지 분할 기준**: `pipeline.py` 상단의 `MAX_SPLIT_HEIGHT`, `SPLIT_OVERLAP` 값 조정
- **작은 이미지 필터 기준**: `MIN_WIDTH`, `MIN_HEIGHT` 값 조정
- **임베딩/분류/추출 모델**: `models.py` 및 각 core 모듈에서 교체 가능

---

## 참고/유의사항

- Selenium 기반이므로 서버에 GUI 없는 환경(headless)에서도 동작
- 크롤링 대상 사이트의 robots.txt 및 이용약관을 반드시 준수하세요
- 대량 크롤링 시 서버 부하 및 차단에 주의

---

## 디렉토리 구조

```
parser_service_updated_prompts/
├── core/
│   ├── web_scraper.py
│   ├── ...
├── models.py
├── pipeline.py
├── main.py
├── config.py
└── ...
```

---
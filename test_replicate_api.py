#!/usr/bin/env python3
"""Replicate API 서버 상태 확인 스크립트"""
import os
import requests
import json
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

if not REPLICATE_API_TOKEN:
    print("❌ REPLICATE_API_TOKEN이 설정되지 않았습니다.")
    exit(1)

print("=" * 60)
print("Replicate API 서버 상태 확인")
print("=" * 60)

# 1. 간단한 모델 목록 조회로 API 연결 테스트
print("\n[1단계] API 연결 테스트 (모델 목록 조회)...")
try:
    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
    }
    # 간단한 GET 요청으로 API 상태 확인
    test_url = "https://api.replicate.com/v1/models/leonardoai/lucid-origin"
    print(f"요청 URL: {test_url}")
    response = requests.get(test_url, headers=headers, timeout=10)
    print(f"응답 상태 코드: {response.status_code}")
    if response.status_code == 200:
        print("✅ API 연결 정상")
        model_info = response.json()
        print(f"모델 정보: {json.dumps(model_info, indent=2, ensure_ascii=False)[:200]}...")
    else:
        print(f"❌ API 연결 실패: {response.status_code}")
        print(f"응답 본문: {response.text[:500]}")
except Exception as e:
    print(f"❌ API 연결 테스트 실패: {e}")

# 2. 실제 이미지 생성 요청 테스트 (간단한 프롬프트)
print("\n[2단계] 이미지 생성 요청 테스트...")
try:
    test_prompt = "a simple red circle on white background"
    request_url = "https://api.replicate.com/v1/models/leonardoai/lucid-origin/predictions"
    body = {
        "input": {
            "prompt": test_prompt,
            "negative_prompt": "text, watermark",
            "aspect_ratio": "16:9",
            "guidance_scale": 3.5,
            "num_inference_steps": 28,
            "image_format": "png"
        }
    }
    print(f"요청 URL: {request_url}")
    print(f"테스트 프롬프트: {test_prompt}")
    response = requests.post(request_url, headers=headers, json=body, timeout=30)
    print(f"응답 상태 코드: {response.status_code}")
    
    if response.status_code in (200, 201):
        print("✅ 이미지 생성 요청 성공")
        prediction = response.json()
        print(f"예측 ID: {prediction.get('id')}")
        print(f"상태: {prediction.get('status')}")
    elif response.status_code >= 500:
        print(f"❌ 서버 에러 ({response.status_code})")
        print(f"응답 본문: {response.text[:500]}")
        print("\n⚠️ Replicate API 서버에 문제가 있습니다.")
        print("   Cloudflare의 Internal Server Error로 인해 서비스가 일시적으로 중단되었을 수 있습니다.")
    else:
        print(f"❌ 요청 실패: {response.status_code}")
        print(f"응답 본문: {response.text[:500]}")
except Exception as e:
    print(f"❌ 이미지 생성 요청 테스트 실패: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("테스트 완료")
print("=" * 60)


import math
import requests
import pandas as pd
from datetime import datetime, timedelta

# 1) 위경도 → 기상청 격자(nx, ny) 변환
def latlon_to_grid(lat, lon):
    RE, GRID = 6371.00877, 5.0
    SLAT1, SLAT2 = 30.0, 60.0
    OLON, OLAT = 126.0, 38.0
    XO, YO = 43, 136

    DEGRAD = math.pi / 180.0
    re = RE / GRID
    slat1, slat2 = SLAT1 * DEGRAD, SLAT2 * DEGRAD
    olon, olat = OLON * DEGRAD, OLAT * DEGRAD

    sn = math.log(math.cos(slat1)/math.cos(slat2)) / \
         math.log(math.tan(math.pi*0.25+slat2*0.5)/math.tan(math.pi*0.25+slat1*0.5))
    sf = (math.tan(math.pi*0.25+slat1*0.5)**sn) * math.cos(slat1) / sn
    ro = re * sf / (math.tan(math.pi*0.25+olat*0.5)**sn)

    ra = re * sf / (math.tan(math.pi*0.25+lat*DEGRAD*0.5)**sn)
    theta = lon*DEGRAD - olon
    theta = (theta + 2*math.pi) % (2*math.pi)
    theta *= sn

    x = int(math.floor(ra*math.sin(theta) + XO + 0.5))
    y = int(math.floor(ro - ra*math.cos(theta) + YO + 0.5))
    return x, y

# 2) 가장 최근 “단기예보” 발표 시각 계산 (3시간 단위)
def get_base_datetime():
    now = datetime.now()
    # 3시간 단위 시각 리스트
    slots = [2, 5, 8, 11, 14, 17, 20, 23]
    # 현재 시간보다 작거나 같은 가장 마지막 slot 찾기
    hour = max([h for h in slots if h <= now.hour] or [23])
    # 자정 이전(0시~1시)이라면 날짜를 하루 전으로
    date = now if now.hour >= 2 else now - timedelta(days=1)

    base = datetime(date.year, date.month, date.day, hour, 0)
    return base.strftime("%Y%m%d"), base.strftime("%H%M")
# 3) 공통 API 호출
SERVICE_KEY = "LnKIdMWHS1qwSq43zdeHQs/v/MOjyfEYU6nncS0rDeM1cYr50AyVPxEwCj7iEbEt1nIUgH3qIxO6tpww+1nx8w=="
BASE_URL = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0"

def fetch_api(endpoint, params):
    url = f"{BASE_URL}/{endpoint}"
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

# 4) 단기예보(getVilageFcst) 조회
def get_vilage_fcst(lat, lon, num_of_rows=1000, page_no=1):
    nx, ny = latlon_to_grid(lat, lon)
    base_date, base_time = get_base_datetime()
    params = {
        "serviceKey": SERVICE_KEY,
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": nx, "ny": ny,
        "numOfRows": num_of_rows, "pageNo": page_no
    }
    data = fetch_api("getVilageFcst", params)

    header = data.get("response", {}).get("header", {})
    if header.get("resultCode") != "00":
        raise RuntimeError(f"API Error {header.get('resultCode')}: {header.get('resultMsg')}")

    return data

# 5) “내일” fcstDate 항목만 추출해 DataFrame 생성
def extract_tomorrow(forecast_json):
    items = forecast_json["response"]["body"]["items"]["item"]  # 이제 body가 보장됨
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")
    rows = []
    for it in items:
        if it["fcstDate"] == tomorrow:
            rows.append({
                "fcstDate": it["fcstDate"],
                "fcstTime": it["fcstTime"],
                "category": it["category"],
                "value": it["fcstValue"]
            })
    return pd.DataFrame(rows)

# 6) 메인 로직: 입력 → 호출 → CSV 저장
def main():
    lat = float(input("위도를 입력하세요: "))
    lon = float(input("경도를 입력하세요: "))

    print("데이터를 불러오는 중입니다...")
    fcst_json = get_vilage_fcst(lat, lon)
    df = extract_tomorrow(fcst_json)

    # ─── 여기에 조회용 코드 추가 ───
    if df.empty:
        print("내일 예보 데이터가 없습니다.")
        return

    print("\n=== 내일 예보 데이터 (DataFrame) ===")
    print(df)  # 또는 df.to_string(index=False) 로 인덱스 없이 보기


    # 파일명 예: forecast_20250506_37.56_126.98.csv
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")
    filename = f"forecast_{tomorrow}_{lat:.2f}_{lon:.2f}.csv"

    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"\n'{filename}' 파일로 저장되었습니다.")

if __name__ == "__main__":
    main()


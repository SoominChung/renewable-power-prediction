import requests
import datetime

# 위도/경도 → 시도명 간단 매핑 예시 (실제 서비스는 더 정교한 매핑 필요)
def get_sido_name(lat, lon):
    # 서울
    if 37.4 <= lat <= 37.7 and 126.7 <= lon <= 127.3:
        return "서울"
    # 경기
    elif 36.5 <= lat <= 38.3 and 126.6 <= lon <= 127.9:
        return "경기"
    # 인천
    elif 37.3 <= lat <= 37.6 and 126.5 <= lon <= 126.8:
        return "인천"
    # 부산
    elif 35.0 <= lat <= 35.3 and 129.0 <= lon <= 129.3:
        return "부산"
    # 대구
    elif 35.7 <= lat <= 36.0 and 128.5 <= lon <= 128.7:
        return "대구"
    # 광주
    elif 35.0 <= lat <= 35.3 and 126.7 <= lon <= 126.9:
        return "광주"
    # 대전
    elif 36.3 <= lat <= 36.5 and 127.3 <= lon <= 127.5:
        return "대전"
    # 울산
    elif 35.4 <= lat <= 35.6 and 129.2 <= lon <= 129.4:
        return "울산"
    # 강원
    elif 37.0 <= lat <= 38.3 and 127.3 <= lon <= 129.3:
        return "강원"
    # 충북
    elif 36.2 <= lat <= 37.2 and 127.3 <= lon <= 128.3:
        return "충북"
    # 충남
    elif 36.0 <= lat <= 36.8 and 126.3 <= lon <= 127.5:
        return "충남"
    # 전북
    elif 35.5 <= lat <= 36.2 and 126.5 <= lon <= 127.5:
        return "전북"
    # 전남
    elif 34.5 <= lat <= 35.5 and 126.2 <= lon <= 127.5:
        return "전남"
    # 경북
    elif 35.6 <= lat <= 36.7 and 128.0 <= lon <= 129.5:
        return "경북"
    # 경남
    elif 34.8 <= lat <= 35.6 and 127.5 <= lon <= 129.3:
        return "경남"
    # 제주
    elif 33.1 <= lat <= 33.6 and 126.1 <= lon <= 126.7:
        return "제주"
    else:
        return None

def get_tomorrow():
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    return tomorrow.strftime("%Y-%m-%d")

def get_pm10_forecast(service_key, search_date):
    url = "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMinuDustFrcstDspth"
    params = {
        "serviceKey": "LnKIdMWHS1qwSq43zdeHQs/v/MOjyfEYU6nncS0rDeM1cYr50AyVPxEwCj7iEbEt1nIUgH3qIxO6tpww+1nx8w==",
        "returnType": "json",
        "searchDate": search_date,
        "InformCode": "PM10"
    }
    response = requests.get(url, params=params)
    return response.json()

def extract_grade(items, sido_name):
    for item in items:
        # 예보일자와 지역명 모두 확인
        grade_info = item.get('informGrade', '')
        if sido_name in grade_info:
            # 예: "서울: 나쁨, 경기: 보통, ... "
            for region_grade in grade_info.split(','):
                if region_grade.strip().startswith(sido_name):
                    return region_grade.strip()
    return None

def main():
    print("위도와 경도를 입력하세요. 예: 37.5665 126.9780")
    try:
        lat = float(input("위도: "))
        lon = float(input("경도: "))
    except ValueError:
        print("숫자로 입력해주세요. 예: 37.5665 126.9780")
        return

    sido_name = get_sido_name(lat, lon)
    if not sido_name:
        print("해당 좌표의 시/도를 찾을 수 없습니다. 주요 도시/지역의 좌표를 입력해주세요.")
        return

    service_key = "여기에_본인_API_KEY_입력"  # 공공데이터포털에서 발급받은 인증키 입력
    tomorrow = get_tomorrow()
    data = get_pm10_forecast(service_key, tomorrow)

    try:
        items = data['response']['body']['items']
        grade = extract_grade(items, sido_name)
        if grade:
            print(f"{tomorrow} {sido_name} 미세먼지(PM10) 예보: {grade}")
        else:
            print(f"{tomorrow} {sido_name}에 대한 예보 데이터를 찾을 수 없습니다.")
    except Exception as e:
        print("미세먼지 예보 데이터를 가져오지 못했습니다.", e)

if __name__ == "__main__":
    main()

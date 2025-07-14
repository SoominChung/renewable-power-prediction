import requests
import base64
import cryptography
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import json
import csv
import xml.etree.ElementTree as ET
import os
import time
from tqdm import tqdm
# encrypt service key with password and save encrypted info as txt file.
def encrypt(filename, file_data, password):
    num_bytes = bytes(password, "utf-8")
    file_data = bytes(file_data, "utf-8")
    salt = bytes(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(num_bytes))
    f = Fernet(key)
    print(type(file_data))
    encrypted_data = f.encrypt(file_data)
    with open(filename, "wb") as file:
        file.write(encrypted_data)
        file.close()
    print("Success encryption for service_key")

# decrypt txt file with password and obtain service key
def decrypt(filename, password):
    num_bytes = bytes(password,"utf-8")
    salt = bytes(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(num_bytes))
    f = Fernet(key)
    with open(filename, "rb") as file:
        # read the encrypted data
        encrypted_data = file.read()
        file.close()
    # decrypt data
    try:
        decrypted_data = f.decrypt(encrypted_data)
    except cryptography.fernet.InvalidToken:
        print("Invalid token, most likely the password is incorrect")
        return
    print("File decrypted successfully\n")
    s_key_str = str(decrypted_data, 'utf-8')
    return s_key_str

# 설정값들 미리 정의
PASSWORD = "0000"
REGIONS = list(input("지역의 이름을 입력하시오: "))
# REGIONS = []
ENERGY_TYPE = 1  # 태양열

# 서비스 키 가져오기
# s_key = decrypt('service_key.txt', PASSWORD)
s_key = 'rGGes+YQRn7qsvorOXy7mhTVGJzkkUV1Mnf6dUIPa4xQVwxZo14PEmBXAGK6ADV9U3l4nWy+zroL/zkUIIZPwA=='
# make_region.py 실행
with open("make_region.py") as f:
    code = f.read()
    exec(code)

# CSV 파일 읽기 (지역 코드)
station_dict = {}
with open("station_data.csv", mode="r", encoding="utf-8") as file:
    reader = csv.reader(file)
    next(reader)  # 첫 번째 행(헤더)은 건너뛰기
    for row in reader:
        station_name, station_code = row
        station_dict[station_name] = station_code

def collect_weather_data(region, start_date, end_date, energy_type, csv_filename):
    """기상 데이터를 수집하는 함수"""
    print(f"수집 중: {region} {start_date} ~ {end_date}")
    
    # 기상 데이터 API 호출
    url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'
    params = {
        'serviceKey': s_key, 
        'pageNo': '1', 
        'numOfRows': '0', 
        'dataType': 'JSON', 
        'dataCd': 'ASOS', 
        'dateCd': 'HR', 
        'startDt': start_date, 
        'startHh': '00', 
        'endDt': end_date, 
        'endHh': '23', 
        'stnIds': station_dict[region]
    }

    response = requests.get(url, params=params)
    response_str = str(response.content, 'utf-8')



    # JSON 문자열 → Python dict
    try:
        response_json = json.loads(response_str)
    except json.JSONDecodeError as e:
        print(response_str)
        # print(response_json)
        print("JSON 파싱 오류 발생:", e)
        return False

    # response_json = json.loads(response_str)
    if response_json['response']['header']['resultCode'] != '00':
        print("!!!! Error messsage !!!!")
        print(str(response_json['response']['header']['resultMsg']))
        return False
    else:
        print("------------- 올바른 기상청데이터를 얻었습니다 ------------------")
    
    data = response.json()  # JSON 파싱

    # 실제 데이터는 response 구조 내에서 'response' → 'body' → 'items' → 'item'에 있음
    items = data['response']['body']['items']['item']
    dates = set()
    
    # 날짜와 시간 분리
    for item in items:
        if 'tm' in item:
            date_part, time_part = item['tm'].split(' ')
            item['date'] = date_part.replace("-", "")
            dates.add(item['date'])
            item['time'] = time_part
            item['temperature'] = item['ta']
            item['humidity'] = item['hm']
            del item['tm'], item['hm'], item['ta']
            if energy_type == 1:  # 태양열
                del item['taQcflg'], item['tsQcflg'], item['rnQcflg'], item['wsQcflg'], item['hmQcflg'], item['td'], item['paQcflg'], item['psQcflg'], item['ssQcflg'], item['dsnw'], item['hr3Fhsc'], item['clfmAbbrCd'], item['gndSttCd'], item['dmstMtphNo'], item['rnum'], item['wdQcflg'], item['m005Te'], item['m01Te'], item['m02Te'], item['m03Te'], item['stnId']

    # 일출, 일몰 가져오기
    url = 'http://apis.data.go.kr/B090041/openapi/service/RiseSetInfoService/getAreaRiseSetInfo'
    sun_data = {}
    print(len(dates))
    i = 0
    check = False
    dates = list(dates) 
    while(i < len(dates)):
        time.sleep(0.5)
        date = dates[i]
        params = {
            'serviceKey': s_key,
            'locdate': date,
            'location': region
        }

        response = requests.get(url, params=params)
        response = requests.get(url, params=params)
        # XML 파====
        root = ET.fromstring(response.content)
        # XML 구조에 따라 item을 순회
        print(i,date)
        for item in root.iter('item'):
            check = True
            locdate = item.findtext('locdate')
            sunrise = item.findtext('sunrise')
            sunset = item.findtext('sunset')
            print(sunrise, sunset)
            if locdate and sunrise and sunset:
                sun_data[locdate] = {'sunrise': sunrise, 'sunset': sunset}
        if check == True:
            i += 1
            check = False
    
    if len(sun_data) != len(dates):
        print(len(sun_data))
        print(response)
        print(root.iter)
        print(root.iter('item'))
        print("!!!! Error messsage !!!!")
        print(" 제대로 정보를 가져오지 못했습니다.\n")
        print(" 프로그램을 종료합니다.")
        return False
    else:
        print("------------- 올바른 일출, 일몰 데이터를 얻었습니다. ------------")

    for item in items:
        if 'date' in item:
            item['sunrise'] = sun_data[item['date']]['sunrise'][:2] + ":" + sun_data[item['date']]['sunrise'][2:]
            item['sunset'] = sun_data[item['date']]['sunset'][:2] + ":" + sun_data[item['date']]['sunset'][2:]

    # 먼저 모든 필드 이름 구함
    all_keys = list(items[0].keys())

    # date와 time을 맨 앞으로 배치하고 나머지 필드는 순서 그대로 추가 (단, 중복 제거)
    fieldnames = ['date', 'time', 'temperature', 'humidity'] + [key for key in all_keys if key not in ['date', 'time', 'temperature', 'humidity']]

    # CSV 저장
    location_path = os.path.join('../../data/weather_data', region)
    if not os.path.exists(location_path):
        os.makedirs(location_path)
    filename = os.path.join(location_path, csv_filename+ ".csv")
    with open(filename, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(items)
        print(f"------------- {csv_filename}.csv 파일을 저장했습니다 ----------------------\n")
    
    return True

# 메인 실행 부분
if __name__ == "__main__":
    # 모든 지역에 대해 데이터 수집
    for region in REGIONS:
        print(f"\n========== {region} 지역 데이터 수집 시작 ==========")
        print("지금")
        start_year = int(input("시작년도: "))
        last_year = int(input("마지막년도: "))
        # 2013년부터 2024년까지 연도별로 데이터 수집
        # for year in range(2013, 2023):
        while (start_year <= last_year):
            start_date = f"{start_year}0101"
            end_date = f"{start_year}1231"
            csv_filename = f"{start_date}_{end_date}"
            
            success = collect_weather_data(region, start_date, end_date, ENERGY_TYPE, csv_filename)
            # time.sleep(60)
            if not success:
                print(f"{region} - {start_year}년 데이터 수집 실패")
                continue
            start_year += 1
        # # 2025년 1월-2월 데이터 수집
        # start_date = "20250101"
        # end_date = "20250228"
        # csv_filename = f"{start_date}_{end_date}"
        
        # success = collect_weather_data(region, start_date, end_date, ENERGY_TYPE, csv_filename)
        # if not success:
        #     print(f"{region} - 2025년 데이터 수집 실패")
        
        print(f"========== {region} 지역 데이터 수집 완료 ==========\n")
    
    print("모든 지역, 모든 연도 데이터 수집이 완료되었습니다!")
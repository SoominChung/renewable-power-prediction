import requests
import base64
import cryptography
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import json
import csv
import xml.etree.ElementTree as ET

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
    print("File decrypted successfully\n\n")
    s_key_str = str(decrypted_data, 'utf-8')
    return s_key_str


res = input("Have you already entered the service key YES: 1, NO: 2 \n")
# password = input("Input the password for obtaining service key: ")
if res == "1":
    password = input("Input the password for obtaining service key: ")
    s_key = decrypt('service_key.txt', password)
elif res == "2":
    s_key = input("Input service key:")
    password = input("Input the password for saving service key: ")
    encrypt('service_key.txt', s_key, password)
else:
    print("Invalid input....")
    exit()

reg = str(input("지역을 입력하시오: "))
s_time = str(input("시작날짜를 입력하시오: "))
e_time = str(input("종료날짜를 입력하시오: "))
select_energy = int(input("원하는 에너지(태양열,풍력)를 입력하시오: "))
csv_name = input("저장할 csv 파일 이름을 입력하시오: ")
# make_region.py 실행
with open("make_region.py") as f:
    code = f.read()
    exec(code)

# CSV 파일 읽기
station_dict = {}

with open("station_data.csv", mode="r", encoding="utf-8") as file:
    reader = csv.reader(file)
    next(reader)  # 첫 번째 행(헤더)은 건너뛰기
    for row in reader:
        station_name, station_code = row
        station_dict[station_name] = station_code


url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'
# numOfRows를 0으로 설정하면 한 번에 찾아내줌(시간은 많이 소요)
# stdIds는 지역번호
params ={'serviceKey' : s_key, 'pageNo' : '1', 'numOfRows' : '0', 'dataType' : 'JSON', 'dataCd' : 'ASOS', 'dateCd' : 'HR', 'startDt' : s_time, 'startHh' : '00', 'endDt' : e_time, 'endHh' : '23', 'stnIds' : station_dict[reg]}

response = requests.get(url, params=params)
response_str = str(response.content, 'utf-8')

# JSON 문자열 → Python dict
response_json = json.loads(response_str)
if response_json['response']['header']['resultCode'] != '00':
    print("!!!! Error messsage !!!!")
    print(str(response_json['response']['header']['resultMsg']))
    exit()
else:
    print("------------- 올바른 기상청데이터를 얻었습니다 ------------------")
    print("------------- 올바른 기상청데이터를 얻었습니다 ------------------\n\n")
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
        if select_energy == 1: # 태양열
            del item['taQcflg'], item['tsQcflg'], item['rnQcflg'], item['wsQcflg'], item['hmQcflg'], item['td'], item['paQcflg'], item['psQcflg'], item['ssQcflg'], item['dsnw'], item['hr3Fhsc'], item['clfmAbbrCd'], item['gndSttCd'], item['dmstMtphNo'], item['rnum'], item['wdQcflg'], item['m005Te'], item['m01Te'], item['m02Te'], item['m03Te'], item['stnId']
        # elif select_energy == 2: # 풍력
        #     del item[]

# 일출, 일몰 가져오기
url = 'http://apis.data.go.kr/B090041/openapi/service/RiseSetInfoService/getAreaRiseSetInfo'
# 날짜, 일출, 일몰 정보 저장할 dict
sun_data = {}
for date in dates:
    params = {
        'serviceKey': s_key,  # 실제 서비스 키로 대체하세요
        'locdate': date,
        'location': reg
    }

    response = requests.get(url, params=params)

    # XML 파싱
    root = ET.fromstring(response.content)

    # XML 구조에 따라 item을 순회
    for item in root.iter('item'):
        locdate = item.findtext('locdate')
        sunrise = item.findtext('sunrise')
        sunset = item.findtext('sunset')

        if locdate and sunrise and sunset:
            sun_data[locdate] = {'sunrise': sunrise, 'sunset': sunset}

if len(sun_data) != len(dates):
    print("!!!! Error messsage !!!!")
    print(" 제대로 정보를 가져오지 못했습니다.\n")
    print(" 프로그램을 종료합니다.")
    exit()
else:
    print("------------- 올바른 일출, 일몰 데이터를 얻었습니다. ------------")
    print("------------- 올바른 일출, 일몰 데이터를 얻었습니다. ------------\n\n")


for item in items:
    if 'date' in item:
        item['sunrise'] = str(sun_data[item['date']]['sunrise'])
        item['sunset'] = str(sun_data[item['date']]['sunset'])

# 먼저 모든 필드 이름 구함
all_keys = list(items[0].keys())

# date와 time을 맨 앞으로 배치하고 나머지 필드는 순서 그대로 추가 (단, 중복 제거)
fieldnames = ['date', 'time', 'temperature', 'humidity'] + [key for key in all_keys if key not in ['date', 'time', 'temperature', 'humidity']]

# CSV 저장
with open(csv_name + ".csv", 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(items)
    print(f"------------- {csv_name}.csv 파일을 저장했습니다 ----------------------")
    print(f"------------- {csv_name}.csv 파일을 저장했습니다 ----------------------\n\n")
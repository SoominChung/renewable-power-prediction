import requests
import base64
import cryptography
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import json
import csv

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
    print("File decrypted successfully")
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

reg_num = str(input("지역번호를 입력하시오: "))
s_time = str(input("시작날짜를 입력하시오: "))
e_time = str(input("종료날짜를 입력하시오: "))
select_energy = int(input("원하는 에너지(태양열,풍력)를 입력하시오: "))


url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'
# numOfRows를 0으로 설정하면 한 번에 찾아내줌(시간은 많이 소요)
# stdIds는 지역번호
params ={'serviceKey' : s_key, 'pageNo' : '1', 'numOfRows' : '0', 'dataType' : 'JSON', 'dataCd' : 'ASOS', 'dateCd' : 'HR', 'startDt' : s_time, 'startHh' : '01', 'endDt' : e_time, 'endHh' : '05', 'stnIds' : reg_num}

response = requests.get(url, params=params)
response_str = str(response.content, 'utf-8')

# JSON 문자열 → Python dict
response_json = json.loads(response_str)
if response_json['response']['header']['resultCode'] != '00':
    print("!!!! Error messsage !!!!")
    print(str(response_json['response']['header']['resultMsg']))
    exit()

data = response.json()  # JSON 파싱

# 실제 데이터는 response 구조 내에서 'response' → 'body' → 'items' → 'item'에 있음
items = data['response']['body']['items']['item']

# 날짜와 시간 분리
for item in items:
    if 'tm' in item:
        date_part, time_part = item['tm'].split(' ')
        item['date'] = date_part
        item['time'] = time_part
        item['temperature'] = item['ta']
        item['humidity'] = item['hm']
        del item['tm'], item['hm'], item['ta']
        if select_energy == 1: # 태양열
            del item['taQcflg'], item['tsQcflg'], item['rnQcflg'], item['wsQcflg'], item['hmQcflg'], item['td'], item['paQcflg'], item['psQcflg'], item['ssQcflg'], item['dsnw'], item['hr3Fhsc'], item['clfmAbbrCd'], item['gndSttCd'], item['dmstMtphNo'], item['rnum'], item['wdQcflg'], item['m005Te'], item['m01Te'], item['m02Te'], item['m03Te'], item['stnId']
        # elif select_energy == 2: # 풍력
        #     del item[]
    
# 먼저 모든 필드 이름 구함
all_keys = list(items[0].keys())

# date와 time을 맨 앞으로 배치하고 나머지 필드는 순서 그대로 추가 (단, 중복 제거)
fieldnames = ['date', 'time', 'temperature', 'humidity'] + [key for key in all_keys if key not in ['date', 'time', 'temperature', 'humidity']]

# CSV 저장
csv_name = input("csv 이름을 입력하시오: ")
with open(csv_name + ".csv", 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(items)
    print("CSV 저장 완료.")

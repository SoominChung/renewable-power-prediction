import requests
import base64
import cryptography
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import json

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
s_time = str(input("시작날짜를 입력하시오 ex) 20200101 \n"))
e_time = str(input("종료날짜를 입력하시오 ex) 20201212 \n"))

url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'
# numOfRows를 0으로 설정하면 한 번에 찾아내줌(시간은 많이 소요)
# stdIds는 지역번호
params ={'serviceKey' : s_key, 'pageNo' : '1', 'numOfRows' : '1', 'dataType' : 'JSON', 'dataCd' : 'ASOS', 'dateCd' : 'HR', 'startDt' : s_time, 'startHh' : '01', 'endDt' : e_time, 'endHh' : '05', 'stnIds' : reg_num}

response = requests.get(url, params=params)
print(response.content)

response_str = str(response.content, 'utf-8')

# 2. JSON 문자열 → Python dict
response_json = json.loads(response_str)
if response_json['response']['header']['resultCode'] != '00':
    print("!!!! Error messsage !!!!")
    print(str(response_json['response']['header']['resultMsg']))
    exit()
# 3. 원하는 값 추출
page_no = response_json['response']['body']['pageNo']
total_count = response_json['response']['body']['totalCount']

print("totalCount:", total_count)
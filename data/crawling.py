import requests
import base64
import cryptography
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


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

url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'
params ={'serviceKey' : s_key, 'pageNo' : '365', 'numOfRows' : '10', 'dataType' : 'JSON', 'dataCd' : 'ASOS', 'dateCd' : 'HR', 'startDt' : '20200101', 'startHh' : '01', 'endDt' : '20200601', 'endHh' : '02', 'stnIds' : '108' }

response = requests.get(url, params=params)
print(response.content)
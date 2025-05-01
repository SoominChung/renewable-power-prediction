import requests
import base64
import cryptography
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

def encrypt(filename, password):
    with open(filename, "rb") as file:
        file_data = file.read()
        file.close()
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
    encrypted_data = f.encrypt(file_data)
    with open(filename, "wb") as file:
        file.write(encrypted_data)
        file.close()
    print("Success encryption for service_key")

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
    with open(filename, "wb") as file:
        file.write(decrypted_data)
        file.close()
    return decrypted_data

password = input("Generate the password for the database: ")
s_key_bytes = decrypt('service_key.txt', password)
encrypt('service_key.txt', password)

s_key_str = str(s_key_bytes, 'utf-8')
url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'
params ={'serviceKey' : s_key_str, 'pageNo' : '365', 'numOfRows' : '10', 'dataType' : 'JSON', 'dataCd' : 'ASOS', 'dateCd' : 'HR', 'startDt' : '20200101', 'startHh' : '01', 'endDt' : '20200601', 'endHh' : '02', 'stnIds' : '108' }

response = requests.get(url, params=params)
print(response.content)

# 데이터 수집
본 페이지는 공공데이터 포털로부터 기상청 데이터수집을 위해 만들었다.

## Execute
service key를 입력한적이 있는지 없는지에따라 나뉘게된다.

### 1. python3 crawling.py
### 2 다음의 상황에따라 번호를 입력한다.
- service_key.txt가 있는 경우 - 1을 입력하고 기존에 등록한 패스워드 입력
- service_key.txt가 없는 경우 - 2을 입력하고 새로운 패스워드 입력
### 3. 지역 정보 입력(아래의 그림에서 가장 근접한 지역의 번호 선택)
![image](https://github.com/SoominChung/renewable-power-prediction/blob/main/data/picture1.png)
### 4. 시작날짜와 종료날짜를 입력한다.
- ex 시작날짜 20200101, 종료날짜 20201231

  ** 단, 날짜를 이상하게 입력하면 에러와함께 프로그램이 종료된다.
### 5. 

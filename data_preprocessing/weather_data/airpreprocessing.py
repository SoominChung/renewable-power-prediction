import pandas as pd
import numpy as np # PM10 등급 분류 시 사용
from google.colab import drive
# drive.mount('/content/drive')

def preprocess_air_quality_data_with_station_name(file_path):
    """
    주어진 경로의 대기질 CSV 파일을 읽어와 요청된 전처리 작업을 수행합니다.
    '측정소명' 열은 유지합니다.

    1. 특정 열('망', '측정소코드', 'SO2', 'CO', 'NO2', '주소') 제거
    2. '지역' 열의 값을 앞 두 글자만 남김
    3. 'PM10' 데이터를 기준으로 'PM10 등급' 열 추가
       - 30 이하: 좋음
       - 31 ~ 80: 보통
       - 81 ~ 150: 나쁨
       - 151 이상: 매우 나쁨
    """
    try:
        # 1. 파일 읽기 (CSV 파일로 가정)
        df = pd.read_excel(file_path)
        print("--- 원본 데이터 (처음 5행) ---")
        print(df.head())
        print(f"\n원본 데이터 모양: {df.shape}")
        # print("\n--- 원본 데이터 정보 ---")
        # df.info() # 상세 정보 필요시 주석 해제
        print("\n")
    except FileNotFoundError:
        return "오류: 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요."
    except Exception as e:
        return f"오류: 파일을 읽는 중 문제가 발생했습니다 - {e}"

    # 2. 특정 열 제거 ('측정소명'은 이 리스트에 포함되지 않음)
    columns_to_drop = ['망', '측정소코드', 'SO2', 'CO', 'NO2', '주소']
    # 실제 파일에 존재하는 열만 제거하도록 필터링
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    if existing_columns_to_drop:
        df = df.drop(columns=existing_columns_to_drop)
        print(f"--- '{', '.join(existing_columns_to_drop)}' 열 제거 후 (처음 5행) ---")
        print(df.head())
        print(f"데이터 모양: {df.shape}")
        print("\n")
    else:
        print(f"제거할 열이 데이터프레임에 없습니다: {columns_to_drop}\n")

    # 3. '지역' 열 수정 (앞 두 글자만 남기기)
    if '지역' in df.columns:
        # 숫자형 데이터가 있을 수 있으므로, 먼저 문자열로 변환
        df['지역'] = df['지역'].astype(str).str[:2]
        print("--- '지역' 열 수정 후 (처음 5행) ---")
        print(df.head())
        print("\n")
    else:
        print("경고: '지역' 열을 찾을 수 없습니다.\n")

    # 4. 'PM10 등급' 열 추가
    if 'PM10' in df.columns:
        # PM10 열을 숫자형으로 변환 (변환 불가능한 값은 NaN으로 처리)
        df['PM10'] = pd.to_numeric(df['PM10'], errors='coerce')

        # 등급 분류 기준 설정
        bins = [-float('inf'), 30, 80, 150, float('inf')]
        labels = ['좋음', '보통', '나쁨', '매우 나쁨']
        
        df['PM10 등급'] = pd.cut(df['PM10'], bins=bins, labels=labels, right=True, include_lowest=True)
        
        # PM10 값이 NaN이라 등급이 NaN인 경우 "알 수 없음" 등으로 채울 수 있습니다.
        # 예: df['PM10 등급'] = df['PM10 등급'].fillna('알 수 없음')

        print("--- 'PM10 등급' 열 추가 후 (처음 5행) ---")
        print(df.head())
        print("\n")
    else:
        print("경고: 'PM10' 열을 찾을 수 없습니다.\n")

    return df

# --- 메인 실행 부분 ---
# 사용자가 업로드한 파일명으로 설정합니다.
# 이 스크립트와 같은 디렉토리에 파일이 있다고 가정합니다.
file_path = '/content/drive/MyDrive/Colab Notebooks/전처리.xlsx'
processed_df_with_station_name = preprocess_air_quality_data_with_station_name(file_path)

if isinstance(processed_df_with_station_name, pd.DataFrame):
    print("--- 최종 처리 결과 (처음 10행, '측정소명' 유지 확인) ---")
    print(processed_df_with_station_name.head(10))
    print(f"\n최종 데이터 모양: {processed_df_with_station_name.shape}")
    
    # 처리된 DataFrame에 '측정소명' 열이 있는지 확인
    if '측정소명' in processed_df_with_station_name.columns:
        print("\n'측정소명' 열이 최종 결과에 포함되어 있습니다.")
    elif '측정소명' in pd.read_csv(file_path).columns: # 원본에는 있었는데 사라진 경우 (이론상 발생 안 함)
        print("\n경고: 원본 파일에는 '측정소명'이 있었으나, 처리 과정에서 누락된 것 같습니다. 코드 확인이 필요합니다.")
    else:
        print("\n참고: 원본 파일에 '측정소명' 열이 없었던 것으로 보입니다.")

    # 처리된 DataFrame을 CSV 파일로 저장하고 싶다면 아래 주석을 해제하세요.
    # output_filename = '처리된_대기질_데이터_측정소명포함.csv'
    # processed_df_with_station_name.to_csv(output_filename, index=False, encoding='utf-8-sig')
    # print(f"\n처리된 데이터가 '{output_filename}' 파일로 저장되었습니다.")
else:
    # 파일 읽기 실패 등의 오류 메시지 출력
    print(processed_df_with_station_name)

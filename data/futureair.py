import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import re # 발표시각에서 시간 부분을 추출하기 위함
import os # 파일 경로 관련 작업을 위해 (선택적이지만, 파일 저장 시 유용)

# --- 설정 ---
SERVICE_KEY = "BeDnNvKEYAxSTHK2tMTlkHzyFpeJEBpnEdF5W6mXPMcne+WmEJcqZkxcAMHP5SzsEqIeW/s63lBuXUzuLahH8w=="
BASE_URL = 'http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMinuDustFrcstDspth'
DEFAULT_INFORM_CODES = ['PM10', 'PM2.5', 'O3'] # 기본 조회 항목

def parse_inform_grade(inform_grade_str, target_region):
    """
    "서울 : 좋음,경기 : 보통,..." 형태의 문자열에서 특정 지역의 등급을 추출합니다.
    """
    if not inform_grade_str or not target_region:
        return None
    try:
        parts = inform_grade_str.split(',')
        for part in parts:
            if ':' in part:
                region, grade = part.split(':', 1)
                if region.strip() == target_region:
                    return grade.strip()
        return None 
    except Exception:
        return None

def get_hour_from_data_time(data_time_str):
    """
    "YYYY-MM-DD HH시 발표" 형식의 문자열에서 HH (시간)을 정수로 추출합니다.
    추출 실패 시 -1 반환.
    """
    if not data_time_str:
        return -1
    match = re.search(r"(\d{1,2})시 발표", data_time_str)
    if match:
        return int(match.group(1))
    return -1

def get_latest_tomorrows_forecast_for_region(service_key, target_region_name, inform_codes_list=None):
    """
    오늘 발표된 예보 중, 내일의 가장 최신 예보를 찾아 PM10, PM2.5, O3 등급을 반환합니다.
    """
    if inform_codes_list is None:
        inform_codes_list = DEFAULT_INFORM_CODES
        
    today_str = datetime.now().strftime('%Y-%m-%d')
    tomorrow_str = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"\n'{target_region_name}' 지역의 내일({tomorrow_str}) 예보 조회 (오늘 {today_str} 발표 기준 최신 정보)")
    
    final_forecast_data = []

    for inform_code in inform_codes_list:
        params = {
            'serviceKey': service_key,
            'returnType': 'json',
            'numOfRows': '100',
            'pageNo': '1',
            'searchDate': today_str,
            'InformCode': inform_code
        }
        
        latest_announcement_for_tomorrow = None
        max_announcement_hour = -1
        
        try:
            response = requests.get(BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if (data.get('response') and 
                data['response'].get('header') and 
                data['response']['header'].get('resultCode') == '00' and
                data['response'].get('body') and 
                data['response']['body'].get('items')):
                
                items = data['response']['body']['items']
                if items:
                    for item_data in items:
                        if item_data.get('informData') == tomorrow_str:
                            announcement_hour = get_hour_from_data_time(item_data.get('dataTime'))
                            if announcement_hour > max_announcement_hour:
                                max_announcement_hour = announcement_hour
                                latest_announcement_for_tomorrow = item_data
                
                if latest_announcement_for_tomorrow:
                    grade = parse_inform_grade(latest_announcement_for_tomorrow.get('informGrade'), target_region_name)
                    final_forecast_data.append({
                        '지역': target_region_name,
                        '항목': inform_code,
                        '예보대상일': tomorrow_str,
                        '예보등급': grade if grade else "정보 없음",
                        '최신발표시각': latest_announcement_for_tomorrow.get('dataTime')
                    })
                else:
                    final_forecast_data.append({
                        '지역': target_region_name,
                        '항목': inform_code,
                        '예보대상일': tomorrow_str,
                        '예보등급': "정보 없음",
                        '최신발표시각': "N/A (해당일 예보 없음)"
                    })
            
            elif data.get('response') and data['response'].get('header'):
                header = data['response']['header']
                print(f"    API 오류 (항목: {inform_code}): {header.get('resultCode')} - {header.get('resultMsg')}")
                final_forecast_data.append({'지역': target_region_name, '항목': inform_code, '예보대상일': tomorrow_str, '예보등급': "API 오류", '최신발표시각': "N/A"})
            else:
                final_forecast_data.append({'지역': target_region_name, '항목': inform_code, '예보대상일': tomorrow_str, '예보등급': "데이터 없음", '최신발표시각': "N/A"})

        except requests.exceptions.RequestException as e:
            print(f"    요청 오류 (항목: {inform_code}): {e}")
            final_forecast_data.append({'지역': target_region_name, '항목': inform_code, '예보대상일': tomorrow_str, '예보등급': "요청 오류", '최신발표시각': "N/A"})
        except json.JSONDecodeError:
             print(f"    JSON 파싱 오류 (항목: {inform_code}). 응답(일부): {response.text[:200] if response else 'N/A'}")
             final_forecast_data.append({'지역': target_region_name, '항목': inform_code, '예보대상일': tomorrow_str, '예보등급': "파싱 오류", '최신발표시각': "N/A"})
        
    if not final_forecast_data:
        print(f"\n{target_region_name} 지역의 내일({tomorrow_str}) 예보 정보를 가져오지 못했습니다.")
        return pd.DataFrame()
        
    df = pd.DataFrame(final_forecast_data)
    return df

# --- 메인 실행 부분 ---
if __name__ == "__main__":
    print("내일의 미세먼지/오존 최신 예보 조회")
    
    input_region_name = input("조회할 지역명 (예: 서울, 부산, 경기 등): ").strip()

    if not input_region_name:
        input_region_name = "서울" # 기본값 설정
        print(f"지역명이 입력되지 않아 기본값 '{input_region_name}'으로 설정합니다.")
    
    forecast_df = get_latest_tomorrows_forecast_for_region(
        SERVICE_KEY,
        input_region_name
    )

    if not forecast_df.empty:
        print(f"\n--- {input_region_name} 지역 내일 예보 (오늘 발표 최신 기준) ---")
        print(forecast_df)

        # --- CSV 파일 저장 로직 추가 ---
        # DataFrame의 '예보대상일'은 모두 동일하므로 첫 번째 행의 값을 사용
        # (만약 DataFrame이 비어있지 않다면 '예보대상일' 컬럼과 해당 값이 존재한다고 가정)
        if '예보대상일' in forecast_df.columns and len(forecast_df['예보대상일']) > 0:
            forecast_date_for_filename = forecast_df['예보대상일'].iloc[0]
            # 파일명에 사용할 수 없는 문자 제거 또는 변경 (예: / \ : * ? " < > |)
            # YYYY-MM-DD 형식은 일반적으로 파일명에 안전합니다.
            csv_filename = f"{forecast_date_for_filename}예보결과.csv"
            
            try:
                # index=False: DataFrame 인덱스를 CSV에 쓰지 않음
                # encoding='utf-8-sig': Excel에서 한글 깨짐 방지
                forecast_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
                print(f"\n데이터를 '{csv_filename}' 파일로 저장했습니다.")
                # Colab 환경이라면 현재 작업 디렉토리에 저장됩니다.
                # print(f"Colab의 경우, 왼쪽 파일 탐색기에서 '{csv_filename}'을 확인하고 다운로드할 수 있습니다.")
            except Exception as e:
                print(f"\nCSV 파일 저장 중 오류 발생: {e}")
        else:
            print("\n경고: '예보대상일' 정보를 찾을 수 없어 CSV 파일명을 생성할 수 없습니다.")
            
    else:
        print(f"\nDataFrame에 표시할 최종 데이터가 없어 CSV 파일로 저장할 수 없습니다.")

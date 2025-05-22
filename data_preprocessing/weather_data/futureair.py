# import requests
# import json
# import pandas as pd
# from datetime import datetime, timedelta
# import re # 발표시각에서 시간 부분을 추출하기 위함
# import os # 파일 경로 관련 작업을 위해 (선택적이지만, 파일 저장 시 유용)

# # --- 설정 ---
# SERVICE_KEY = "BeDnNvKEYAxSTHK2tMTlkHzyFpeJEBpnEdF5W6mXPMcne+WmEJcqZkxcAMHP5SzsEqIeW/s63lBuXUzuLahH8w=="
# BASE_URL = 'http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMinuDustFrcstDspth'
# DEFAULT_INFORM_CODES = ['PM10', 'PM2.5', 'O3'] # 기본 조회 항목

# def parse_inform_grade(inform_grade_str, target_region):
#     """
#     "서울 : 좋음,경기 : 보통,..." 형태의 문자열에서 특정 지역의 등급을 추출합니다.
#     """
#     if not inform_grade_str or not target_region:
#         return None
#     try:
#         parts = inform_grade_str.split(',')
#         for part in parts:
#             if ':' in part:
#                 region, grade = part.split(':', 1)
#                 if region.strip() == target_region:
#                     return grade.strip()
#         return None 
#     except Exception:
#         return None

# def get_hour_from_data_time(data_time_str):
#     """
#     "YYYY-MM-DD HH시 발표" 형식의 문자열에서 HH (시간)을 정수로 추출합니다.
#     추출 실패 시 -1 반환.
#     """
#     if not data_time_str:
#         return -1
#     match = re.search(r"(\d{1,2})시 발표", data_time_str)
#     if match:
#         return int(match.group(1))
#     return -1

# def get_latest_tomorrows_forecast_for_region(service_key, target_region_name, inform_codes_list=None):
#     """
#     오늘 발표된 예보 중, 내일의 가장 최신 예보를 찾아 PM10, PM2.5, O3 등급을 반환합니다.
#     """
#     if inform_codes_list is None:
#         inform_codes_list = DEFAULT_INFORM_CODES
        
#     today_str = datetime.now().strftime('%Y-%m-%d')
#     tomorrow_str = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
#     print(f"\n'{target_region_name}' 지역의 내일({tomorrow_str}) 예보 조회 (오늘 {today_str} 발표 기준 최신 정보)")
    
#     final_forecast_data = []

#     for inform_code in inform_codes_list:
#         params = {
#             'serviceKey': service_key,
#             'returnType': 'json',
#             'numOfRows': '100',
#             'pageNo': '1',
#             'searchDate': today_str,
#             'InformCode': inform_code
#         }
        
#         latest_announcement_for_tomorrow = None
#         max_announcement_hour = -1
        
#         try:
#             response = requests.get(BASE_URL, params=params, timeout=10)
#             response.raise_for_status()
#             data = response.json()

#             if (data.get('response') and 
#                 data['response'].get('header') and 
#                 data['response']['header'].get('resultCode') == '00' and
#                 data['response'].get('body') and 
#                 data['response']['body'].get('items')):
                
#                 items = data['response']['body']['items']
#                 if items:
#                     for item_data in items:
#                         if item_data.get('informData') == tomorrow_str:
#                             announcement_hour = get_hour_from_data_time(item_data.get('dataTime'))
#                             if announcement_hour > max_announcement_hour:
#                                 max_announcement_hour = announcement_hour
#                                 latest_announcement_for_tomorrow = item_data
                
#                 if latest_announcement_for_tomorrow:
#                     grade = parse_inform_grade(latest_announcement_for_tomorrow.get('informGrade'), target_region_name)
#                     final_forecast_data.append({
#                         '지역': target_region_name,
#                         '항목': inform_code,
#                         '예보대상일': tomorrow_str,
#                         '예보등급': grade if grade else "정보 없음",
#                         '최신발표시각': latest_announcement_for_tomorrow.get('dataTime')
#                     })
#                 else:
#                     final_forecast_data.append({
#                         '지역': target_region_name,
#                         '항목': inform_code,
#                         '예보대상일': tomorrow_str,
#                         '예보등급': "정보 없음",
#                         '최신발표시각': "N/A (해당일 예보 없음)"
#                     })
            
#             elif data.get('response') and data['response'].get('header'):
#                 header = data['response']['header']
#                 print(f"    API 오류 (항목: {inform_code}): {header.get('resultCode')} - {header.get('resultMsg')}")
#                 final_forecast_data.append({'지역': target_region_name, '항목': inform_code, '예보대상일': tomorrow_str, '예보등급': "API 오류", '최신발표시각': "N/A"})
#             else:
#                 final_forecast_data.append({'지역': target_region_name, '항목': inform_code, '예보대상일': tomorrow_str, '예보등급': "데이터 없음", '최신발표시각': "N/A"})

#         except requests.exceptions.RequestException as e:
#             print(f"    요청 오류 (항목: {inform_code}): {e}")
#             final_forecast_data.append({'지역': target_region_name, '항목': inform_code, '예보대상일': tomorrow_str, '예보등급': "요청 오류", '최신발표시각': "N/A"})
#         except json.JSONDecodeError:
#              print(f"    JSON 파싱 오류 (항목: {inform_code}). 응답(일부): {response.text[:200] if response else 'N/A'}")
#              final_forecast_data.append({'지역': target_region_name, '항목': inform_code, '예보대상일': tomorrow_str, '예보등급': "파싱 오류", '최신발표시각': "N/A"})
        
#     if not final_forecast_data:
#         print(f"\n{target_region_name} 지역의 내일({tomorrow_str}) 예보 정보를 가져오지 못했습니다.")
#         return pd.DataFrame()
        
#     df = pd.DataFrame(final_forecast_data)
#     return df

# # --- 메인 실행 부분 ---
# if __name__ == "__main__":
#     print("내일의 미세먼지/오존 최신 예보 조회")
    
#     input_region_name = input("조회할 지역명 (예: 서울, 부산, 경기 등): ").strip()

#     if not input_region_name:
#         input_region_name = "서울" # 기본값 설정
#         print(f"지역명이 입력되지 않아 기본값 '{input_region_name}'으로 설정합니다.")
    
#     forecast_df = get_latest_tomorrows_forecast_for_region(
#         SERVICE_KEY,
#         input_region_name
#     )

#     if not forecast_df.empty:
#         print(f"\n--- {input_region_name} 지역 내일 예보 (오늘 발표 최신 기준) ---")
#         print(forecast_df)

#         # --- CSV 파일 저장 로직 추가 ---
#         # DataFrame의 '예보대상일'은 모두 동일하므로 첫 번째 행의 값을 사용
#         # (만약 DataFrame이 비어있지 않다면 '예보대상일' 컬럼과 해당 값이 존재한다고 가정)
#         if '예보대상일' in forecast_df.columns and len(forecast_df['예보대상일']) > 0:
#             forecast_date_for_filename = forecast_df['예보대상일'].iloc[0]
#             # 파일명에 사용할 수 없는 문자 제거 또는 변경 (예: / \ : * ? " < > |)
#             # YYYY-MM-DD 형식은 일반적으로 파일명에 안전합니다.
#             csv_filename = f"{forecast_date_for_filename}예보결과.csv"
            
#             try:
#                 # index=False: DataFrame 인덱스를 CSV에 쓰지 않음
#                 # encoding='utf-8-sig': Excel에서 한글 깨짐 방지
#                 forecast_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
#                 print(f"\n데이터를 '{csv_filename}' 파일로 저장했습니다.")
#                 # Colab 환경이라면 현재 작업 디렉토리에 저장됩니다.
#                 # print(f"Colab의 경우, 왼쪽 파일 탐색기에서 '{csv_filename}'을 확인하고 다운로드할 수 있습니다.")
#             except Exception as e:
#                 print(f"\nCSV 파일 저장 중 오류 발생: {e}")
#         else:
#             print("\n경고: '예보대상일' 정보를 찾을 수 없어 CSV 파일명을 생성할 수 없습니다.")
            
#     else:
#         print(f"\nDataFrame에 표시할 최종 데이터가 없어 CSV 파일로 저장할 수 없습니다.")

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
        current_error_message = "정보 없음" # 각 항목별 오류 메시지 초기화

        try:
            print(f"\n[요청 시작] 항목: {inform_code}, 지역: {target_region_name}") # 요청 정보 출력
            response = requests.get(BASE_URL, params=params, timeout=15) # 타임아웃 약간 늘림

            # --- 👇 중요: 응답 내용 직접 확인 코드 추가 👇 ---
            print(f"[응답 수신] 항목: {inform_code}, HTTP 상태 코드: {response.status_code}")
            # 실제 응답 텍스트의 앞부분 300자 출력 (디버깅 시 매우 유용)
            print(f"[응답 내용(일부)] 항목: {inform_code}: {response.text[:300]}")
            # --- 👆 중요: 응답 내용 직접 확인 코드 추가 👆 ---

            response.raise_for_status() # HTTP 오류 발생 시 예외 처리 (200 OK 아니면 여기서 에러 발생)
            data = response.json() # 여기서 JSONDecodeError 발생 가능

            if (data.get('response') and
                data['response'].get('header') and
                data['response']['header'].get('resultCode') == '00' and
                data['response'].get('body') and
                data['response']['body'].get('items')):

                items = data['response']['body']['items']
                if items: # items가 비어있지 않은 경우
                    for item_data in items:
                        if item_data.get('informData') == tomorrow_str:
                            announcement_hour = get_hour_from_data_time(item_data.get('dataTime'))
                            if announcement_hour > max_announcement_hour:
                                max_announcement_hour = announcement_hour
                                latest_announcement_for_tomorrow = item_data
                else: # items가 비어있는 경우 (정상 응답이나 데이터가 없는 경우)
                    current_error_message = "해당일 예보 없음 (items 비어 있음)"


                if latest_announcement_for_tomorrow:
                    grade = parse_inform_grade(latest_announcement_for_tomorrow.get('informGrade'), target_region_name)
                    final_forecast_data.append({
                        '지역': target_region_name,
                        '항목': inform_code,
                        '예보대상일': tomorrow_str,
                        '예보등급': grade if grade else "등급 정보 없음",
                        '최신발표시각': latest_announcement_for_tomorrow.get('dataTime')
                    })
                    continue # 다음 inform_code로 성공적으로 넘어감
                else: # latest_announcement_for_tomorrow가 None인 경우 (루프는 돌았으나 조건 맞는 데이터 없음)
                    # current_error_message가 "해당일 예보 없음 (items 비어 있음)"이거나 기본값 "정보 없음"일 수 있음
                    if current_error_message == "정보 없음": # items는 있었으나 내일자 정보가 없는 경우
                         current_error_message = "내일자 예보 없음"

            # API 응답은 성공(200)이나 resultCode가 '00'이 아닌 경우
            elif data.get('response') and data['response'].get('header'):
                header = data['response']['header']
                current_error_message = f"API 오류: {header.get('resultCode')} - {header.get('resultMsg')}"
                print(f"  항목 {inform_code} API 응답 메시지: {current_error_message}")
            # 예상치 못한 JSON 구조
            else:
                current_error_message = "데이터 구조 이상"
                print(f"  항목 {inform_code} 데이터 구조 이상. 응답(일부): {response.text[:200]}")

        # 예외 처리 블록 상세화
        except requests.exceptions.HTTPError as e:
            current_error_message = f"HTTP 오류: {e}"
            # response 객체가 있을 경우에만 text 접근
            print(f"  요청 오류 (항목: {inform_code}): {current_error_message}. 응답 내용: {response.text[:200] if response else 'N/A'}")
        except requests.exceptions.ConnectionError as e:
            current_error_message = f"연결 오류: {e}"
            print(f"  요청 오류 (항목: {inform_code}): {current_error_message}")
        except requests.exceptions.Timeout as e:
            current_error_message = f"타임아웃: {e}"
            print(f"  요청 오류 (항목: {inform_code}): {current_error_message}")
        except requests.exceptions.TooManyRedirects as e:
            current_error_message = f"리디렉션 초과: {e}"
            print(f"  요청 오류 (항목: {inform_code}): {current_error_message}")
        except json.JSONDecodeError:
            # response.text는 이미 위에서 출력 시도했으므로, 여기서는 오류 메시지만 명시
            current_error_message = "JSON 파싱 오류"
            print(f"  JSON 파싱 오류 (항목: {inform_code}). HTTP 상태: {response.status_code if response else 'N/A'}. 이미 출력된 응답 내용을 확인하세요.")
        except requests.exceptions.RequestException as e: # 기타 requests 관련 오류
            current_error_message = f"일반 요청 오류: {e}"
            print(f"  요청 오류 (항목: {inform_code}): {current_error_message}")
        except Exception as e: # 그 외 모든 예외
            current_error_message = f"기타 예외: {str(e)}"
            print(f"  처리 중 예외 발생 (항목: {inform_code}): {current_error_message}")


        # for 루프의 마지막: 성공적으로 데이터를 찾았거나, 어떤 종류의 오류든 발생했거나, 정보가 없는 경우
        # latest_announcement_for_tomorrow가 설정되지 않았다면 (즉, 위에서 continue로 빠져나가지 못했다면)
        # 여기에 해당 항목의 결과를 추가 (오류 정보 포함)
        found_data_for_current_inform_code = any(
            d['항목'] == inform_code and d['예보등급'] not in ["요청 오류", "API 오류", "데이터 없음", "파싱 오류", "정보 없음", current_error_message]
            for d in final_forecast_data
        )

        if not found_data_for_current_inform_code:
            final_forecast_data.append({
                '지역': target_region_name,
                '항목': inform_code,
                '예보대상일': tomorrow_str,
                '예보등급': current_error_message, # 가장 마지막으로 설정된 오류/상태 메시지
                '최신발표시각': "N/A"
            })


    if not final_forecast_data:
        print(f"\n{target_region_name} 지역의 내일({tomorrow_str}) 예보 정보를 가져오지 못했습니다.")
        return pd.DataFrame()

    df = pd.DataFrame(final_forecast_data)
    return df

# --- 메인 실행 부분 ---
if __name__ == "__main__":
    print("내일의 미세먼지/오존 최신 예보 조회")

    # ⚠️ 서비스 키 관리 주의: 실제 키로 교체하고, 코드에 직접 노출하지 마세요.
    # 환경 변수 사용을 강력히 권장합니다.
    # 예: SERVICE_KEY = os.environ.get("AIR_KOREA_API_KEY")
    if SERVICE_KEY == "YOUR_ACTUAL_DECODED_SERVICE_KEY" or SERVICE_KEY == "BeDnNvKEYAxSTHK2tMTlkHzyFpeJEBpnEdF5W6mXPMcne+WmEJcqZkxcAMHP5SzsEqIeW/s63lBuXUzuLahH8w==":
        print("\n🚨 경고: SERVICE_KEY를 실제 유효한 '디코딩된' 서비스 키로 변경해야 합니다.")
        print("현재 설정된 키로는 API 호출이 실패할 가능성이 매우 높습니다.")
        # 실제 사용 시에는 여기서 프로그램을 중단하거나, 사용자에게 키 입력을 다시 받는 것이 좋습니다.
        # exit() # 필요시 주석 해제하여 프로그램 중단

    input_region_name = input("조회할 지역명 (예: 서울, 부산, 경기 등): ").strip()

    if not input_region_name:
        input_region_name = "서울"
        print(f"지역명이 입력되지 않아 기본값 '{input_region_name}'으로 설정합니다.")

    forecast_df = get_latest_tomorrows_forecast_for_region(
        SERVICE_KEY,
        input_region_name
    )

    if not forecast_df.empty:
        print(f"\n--- {input_region_name} 지역 내일 예보 (오늘 발표 최신 기준) ---")
        print(forecast_df)

        # CSV 저장 로직은 주석 처리된 상태로 유지
        # if '예보대상일' in forecast_df.columns and len(forecast_df['예보대상일']) > 0:
        #     # ... (CSV 저장 코드)
        # else:
        #     print("\n경고: '예보대상일' 정보를 찾을 수 없어 CSV 파일명을 생성할 수 없습니다.")
    else:
        # 이 메시지는 get_latest_tomorrows_forecast_for_region 함수 내부에서도 출력될 수 있습니다.
        # 필요에 따라 중복 출력을 조정하거나, 여기서만 출력하도록 변경 가능합니다.
        print(f"\n'{input_region_name}' 지역에 대한 최종 예보 데이터를 가져오지 못했습니다.")

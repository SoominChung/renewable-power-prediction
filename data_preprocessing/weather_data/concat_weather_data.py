import os
import pandas as pd
import re

def concat_csvs_by_region_and_year(base_dir, region, start_year, end_year, output_dir):
    region_dir = os.path.join(base_dir, region)
    matched_files = []

    # 폴더가 존재하지 않으면 종료
    if not os.path.isdir(region_dir):
        print(f"[ERROR] 지역 폴더가 존재하지 않습니다: {region_dir}")
        return
    
    # 파일 필터링
    for file in os.listdir(region_dir):
        if file.endswith(".csv"):
            match = re.match(r"(\d{8})_(\d{8})\.csv", file)
            if match:
                start_date = match.group(1)
                year = int(start_date[:4])

                if start_year <= year <= end_year:
                    full_path = os.path.join(region_dir, file)
                    matched_files.append(full_path)

    if not matched_files:
        print(f"[INFO] 해당 연도 범위에 맞는 파일이 없습니다: {region}")
        return
    matched_files.sort()
    # CSV 병합
    df_list = [pd.read_csv(f) for f in matched_files]
    combined_df = pd.concat(df_list, ignore_index=True)

    # 저장할 디렉토리 없으면 생성
    os.makedirs(output_dir, exist_ok=True)

    # 파일명 지정: 예) 충주2013_2021.csv
    output_filename = f"{region}{start_year}_{end_year}.parquet"
    output_path = os.path.join(output_dir, output_filename)

    # 날짜 column을 문자열로 변환 후 형식 변경
    combined_df['date'] = pd.to_datetime(combined_df['date'].astype(str), format='%Y%m%d').dt.strftime('%Y-%m-%d')    
    # CSV 저장
    # combined_df.to_parquet(output_path, index=False)
    # combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    combined_df.to_parquet(output_path, index=False, engine='pyarrow')
    
    print(f"[OK] 저장 완료: {output_path}")

if __name__ == "__main__":
    # base_dir = os.getcwd()    # 예: 현재 디렉토리
    base_dir = os.path.abspath("../../data/weather_data")
    output_dir = os.path.join(base_dir, "preprocessed_data")  # 결과 저장 경로
    print(base_dir)
    start_year = int(input("시작년도: "))
    end_year = int(input("마지막년도: "))
    
    # 하위 항목 중 디렉토리인 것만 필터링
    regions = [name for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name))]
    if "preprocessed_data" in regions:
        regions.remove("preprocessed_data")
      # 필요한 지역들

    for region in regions:
        concat_csvs_by_region_and_year(base_dir, region, start_year, end_year, output_dir)

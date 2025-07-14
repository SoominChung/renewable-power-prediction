import os
import pandas as pd
from datetime import datetime, timedelta
import re

root_path = '../../data/weather_data_forecast'

kkkk = []
for region in os.listdir(root_path):
    region_path = os.path.join(root_path, region)
    if not os.path.isdir(region_path):
        continue
    t = []
    
    for year in os.listdir(region_path):
        year_path = os.path.join(region_path, year)
        l = []
        if not os.path.isdir(year_path):
            continue

        for filename in os.listdir(year_path):
            if not filename.endswith('.csv'):
                continue

            file_path = os.path.join(year_path, filename)
            print(f"처리중: {file_path}")

            # 파일명에서 컬럼명과 시작월 추출
            parts = filename.replace('.csv', '').split('_')
            if len(parts) < 3:
                print(f"파일명 포맷 오류, 건너뜀: {filename}")
                continue

            keyword = parts[1]
            start_ym = parts[2]
            try:
                current_base_date = datetime.strptime(start_ym , '%Y%m%d')
            except Exception as e:
                print(f"시작월 파싱 실패 {start_ym}: {e}")
                continue

            try:
                df_raw = pd.read_csv(file_path, header=None)
                df_raw.columns = ['day', 'hour', 'forecast', keyword]
            except Exception as e:
                print(f"파일 읽기 실패 {file_path}: {e}")
                continue

            processed_rows = []

            for idx, row in df_raw.iterrows():
                day_val = row['day']

                if isinstance(day_val, str):
                    day_val_clean = day_val.lower().replace(" ", "")
                    if day_val_clean.startswith('start:'):
                        m = re.match(r'start:(\d{8})', day_val_clean)
                        if m:
                            date_str = m.group(1)
                            current_base_date = datetime.strptime(date_str, '%Y%m%d')
                            continue

                try:
                    day_int = int(day_val)
                except:
                    continue
                new_row = row.copy()
                new_row['day'] = current_base_date.strftime('%Y%m%d')
                processed_rows.append(new_row)

            df_processed = pd.DataFrame(processed_rows)
            l.append(df_processed)

            # 1. 가장 row가 긴 df 기준 선택
        l = sorted(l, key=lambda df: len(df), reverse=True)
        base_df = l[0]
        merged_df = base_df[['day', 'hour', 'forecast']].copy()
        base_colname = base_df.columns[3]
        merged_df[base_colname] = base_df[base_colname]

        # 2. 나머지 병합
        for df in l[1:]:
            col_name = df.columns[3]
            df_renamed = df[['day', 'hour', 'forecast', col_name]].copy()

            merged_df = pd.merge(
                merged_df,
                df_renamed,
                on=['day', 'hour', 'forecast'],
                how='outer'
            )
            print(f"✅ 병합 완료: {col_name}")
        
        # 3. forecast 정수 변환 및 정렬
        merged_df['forecast'] = pd.to_numeric(merged_df['forecast'], errors='coerce')
        merged_df = merged_df.sort_values(by=['day', 'hour', 'forecast'], ascending=[True, True, True])
        t.append(merged_df)
    for i in range(len(t)):
        # print(len[t[i]])
        if i == 0:
            df_combined = t[0]
        else:
            df_combined = pd.concat([df_combined, t[i]], ignore_index=True)
    df_combined = df_combined.sort_values(by=['day', 'hour', 'forecast'], ascending=[True, True, True])
    # df_combined = pd.concat([df1, df2], ignore_index=True)
    # 4. 저장 경로 지정
    df_combined['hour'] = df_combined['hour'].astype(int)
    df_combined['hour'] = df_combined['hour'].fillna('').astype(str)
    df_combined['forecast'] = df_combined['forecast'].astype(int)
    df_combined['forecast'] = df_combined['forecast'].fillna('').astype(str)
    for i in range(5):
        df_combined[df_combined.columns[i+3]] = df_combined[df_combined.columns[i]].fillna('').astype(str)
    
    output_dir = '../../data/total_data'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, region +  '_prediction_weather_data.parquet')
    
    # 5. 저장
    df_combined.to_parquet(output_path, index=False, engine='pyarrow')
    
    # df_combined.to_csv(output_path, index=False)
    print(f"\n✅ 병합 파일 저장 완료: {output_path}")

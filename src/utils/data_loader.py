import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import seaborn as sns
import platform



class PortDataLoader:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.data: Optional[pd.DataFrame] = None
        self.file_path = self.data_dir / "master_merged.csv"

    def load_data(self) -> None:
        """병합된 CSV 데이터 로드"""
        print("\n=== 데이터 로드 시작 ===")
        
        try:
            if not self.file_path.exists():
                print(f"파일을 찾을 수 없음: {self.file_path}")
                return
            
            # CSV 파일 로드
            print(f"\n로딩 중: {self.file_path.name}")
            self.data = pd.read_csv(self.file_path)
            
            # 데이터 타입 최적화
            self.data = self._optimize_dtypes(self.data)
            
            print(f"로드 완료 (shape: {self.data.shape})")
            print(f"메모리 사용량: {self.data.memory_usage().sum() / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            print(f"데이터 로드 중 에러 발생: {str(e)}")
            self.data = None

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터프레임의 메모리 사용량 최적화"""
        for col in df.columns:
            if df[col].dtype == 'object':
                if df[col].nunique() / len(df[col]) < 0.5:  # 카테고리형 데이터
                    df[col] = df[col].astype('category')
            elif df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                df[col] = df[col].astype('int32')
        return df  # 누락된 return 문 추가

    def analyze_block_distribution(self) -> dict:
        if self.data is None:
            print("데이터가 로드되지 않았습니다. load_data()를 먼저 실행하세요.")
            return {}
        
        results = {
            "전체 데이터 현황": {
                "총 레코드 수": len(self.data),
                "분석 기간": f"{self.data['OUT_DATE'].min()} ~ {self.data['OUT_DATE'].max()}"
            },
            "블록 현황": {
                "총 블록 수": self.data['BLOCK'].nunique(),
                "블록별 물동량": self.data.groupby('BLOCK').size().to_dict()
            }
        }
        
        # 블록별 주요 화물 유형
        block_cargo_analysis = self.data.groupby(['BLOCK', 'CARGO_TYPE']).size().reset_index(name='count')
        top_cargo_by_block = block_cargo_analysis.sort_values('count', ascending=False).groupby('BLOCK').first()
        results["블록별 주요 화물"] = top_cargo_by_block.to_dict('index')

        return results
    
    def visualize_block_distribution(self):
        """블록별 물동량 분포를 시각화"""
        if self.data is None:
            print("데이터가 로드되지 않았습니다. load_data()를 먼저 실행하세요.")
            return
        
        # 운영체제별 한글 폰트 설정
        if platform.system() == 'Windows':
            font_path = "C:/Windows/Fonts/malgun.ttf"   # 윈도우의 맑은 고딕 폰트 경로
            font = font_manager.FontProperties(fname=font_path).get_name()
            rc('font', family=font)
        else:    # Mac이나 Linux의 경우
            rc('font', family='AppleGothic')
        
        plt.rcParams['axes.unicode_minus'] = False     # 마이너스 기호 깨짐 방지
        
        # 그래프 생성
        plt.figure(figsize=(15, 6))
        
        # 블록별 물동량 계산
        block_volumes = self.data['BLOCK'].value_counts().sort_index()
        
        # 바 차트 생성
        ax = block_volumes.plot(kind='bar', color='skyblue')
        plt.title('블록별 물동량 분포', pad=20, fontsize=14)
        plt.xlabel('블록 번호', fontsize=12)
        plt.ylabel('물동량 (건수)', fontsize=12)
        
        # x축 레이블 회전
        plt.xticks(rotation=45, ha='right')
        
        # 그리드 추가
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 바 위에 값 표시
        for i, v in enumerate(block_volumes):
            ax.text(i, v, format(v, ','), ha='center', va='bottom')
        
        # 그래프 레이아웃 조정
        plt.tight_layout()
        
        # 그래프 표시
        plt.show()
    
    def print_analysis_results(self, results: dict) -> None:
        """분석 결과 출력"""
        print("\n=== 블록 분포 분석 결과 ===\n")
        
        if not results:
            print("분석 결과가 없습니다.")
            return
            
        for key, value in results.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  - {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
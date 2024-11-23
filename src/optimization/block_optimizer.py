import pandas as pd
import numpy as np
from typing import Dict, List
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numba

class BlockOptimizer:
    def __init__(self, data: pd.DataFrame, sample_size: int = 10000):
        if len(data) > sample_size:
            self.data = data.sample(n=sample_size, random_state=42)
        else:
            self.data = data
            
        # 데이터 전처리
        self.data['IN_DATE'] = pd.to_datetime(self.data['IN_DATE'])
        self.data['OUT_DATE'] = pd.to_datetime(self.data['OUT_DATE'])
        self.data['initial_TAT'] = (self.data['OUT_DATE'] - self.data['IN_DATE']).dt.total_seconds() / 60
        
        self.cargo_volumes = self.data['CARGO_TYPE'].value_counts()
        self.cargo_types = self.cargo_volumes.index.tolist()
        self.current_blocks = sorted(self.data['BLOCK'].unique())
        self.total_blocks = len(self.current_blocks)
        
        print(f"시뮬레이션 데이터 크기: {len(self.data):,}개 레코드")

    def calculate_tat(self, block_allocation: Dict[str, List[str]]) -> float:
        """TAT 계산"""
        simulation_data = self.data.copy()
        
        for cargo_type, blocks in block_allocation.items():
            mask = simulation_data['CARGO_TYPE'] == cargo_type
            if not mask.any():
                continue
            
            # 현재 블록과 새로운 블록 간의 거리 계산
            current_blocks = simulation_data.loc[mask, 'BLOCK'].values
            # 각 화물에 대해 랜덤하게 새 블록 할당
            new_blocks = np.random.choice(blocks, size=len(current_blocks))
            
            # 블록 인덱스 기반 거리 계산
            current_indices = [self.current_blocks.index(b) for b in current_blocks]
            new_indices = [self.current_blocks.index(b) for b in new_blocks]
            block_distances = np.abs(np.array(current_indices) - np.array(new_indices))
            
            # 처리량 기반 가중치
            volume_factor = 1 if len(blocks) > 5 else 2
            time_adjustments = np.minimum(block_distances * volume_factor, 10)
            
            # OUT_DATE 조정
            simulation_data.loc[mask, 'OUT_DATE'] = (
                simulation_data.loc[mask, 'OUT_DATE'] - 
                pd.to_timedelta(time_adjustments, unit='minutes')
            )
        
        # 최종 TAT 계산
        simulation_data['TAT'] = (
            simulation_data['OUT_DATE'] - simulation_data['IN_DATE']
        ).dt.total_seconds() / 60
        
        return simulation_data['TAT'].mean()

    def simulate_block_ratio(self, num_simulations: int = 100) -> Dict:
        """블록 비율 시뮬레이션"""
        best_allocation = None
        best_tat = float('inf')
        results = []
        
        volume_weights = self.cargo_volumes / self.cargo_volumes.sum()
        base_blocks = (volume_weights * self.total_blocks).round()
        
        for _ in range(num_simulations):
            # 블록 수 할당
            variation = np.random.uniform(0.8, 1.2, len(self.cargo_types))
            adjusted_blocks = base_blocks * variation
            
            block_counts = {
                cargo_type: max(1, int(blocks))
                for cargo_type, blocks in zip(self.cargo_types, adjusted_blocks)
            }
            
            # 전체 블록 수 조정
            while sum(block_counts.values()) != self.total_blocks:
                if sum(block_counts.values()) > self.total_blocks:
                    cargo_type = max(block_counts, key=block_counts.get)
                    if block_counts[cargo_type] > 1:
                        block_counts[cargo_type] -= 1
                else:
                    cargo_type = max(self.cargo_types,
                                   key=lambda x: self.cargo_volumes[x]/block_counts[x])
                    block_counts[cargo_type] += 1
            
            # 블록 할당
            available_blocks = self.current_blocks.copy()
            block_allocation = {}
            
            for cargo_type in sorted(self.cargo_types, 
                                   key=lambda x: self.cargo_volumes[x],
                                   reverse=True):
                count = block_counts[cargo_type]
                allocated = available_blocks[:count]
                block_allocation[cargo_type] = allocated
                available_blocks = available_blocks[count:]
            
            # TAT 계산
            current_tat = self.calculate_tat(block_allocation)
            
            if current_tat < best_tat:
                best_tat = current_tat
                best_allocation = block_allocation
            
            results.append({
                'allocation': block_allocation,
                'tat': current_tat,
                'block_counts': block_counts
            })
        
        return {
            'best_allocation': best_allocation,
            'best_tat': best_tat,
            'all_results': sorted(results, key=lambda x: x['tat'])
        }
    
    def analyze_results(self, simulation_results: Dict) -> Dict:
        """
        시뮬레이션 결과 분석
        """
        best_allocation = simulation_results['best_allocation']
        best_tat = simulation_results['best_tat']
        
        # 현재 블록 할당 분석
        current_allocation = {
            cargo_type: self.data[self.data['CARGO_TYPE'] == cargo_type]['BLOCK'].unique().tolist()
            for cargo_type in self.cargo_types
        }
        current_tat = self.calculate_tat(current_allocation)
        
        return {
            'current_tat': current_tat,
            'optimized_tat': best_tat,
            'improvement': ((current_tat - best_tat) / current_tat) * 100,
            'current_allocation': current_allocation,
            'optimized_allocation': best_allocation,
            'block_counts': {
                'current': {ct: len(blocks) for ct, blocks in current_allocation.items()},
                'optimized': {ct: len(blocks) for ct, blocks in best_allocation.items()}
            }
        }
    def visualize_results(self, analysis: Dict):
        """시뮬레이션 결과 시각화"""
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

        # 1. TAT 비교와 블록 할당 비교
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # TAT 비교 막대 그래프
        tat_comparison = [analysis['current_tat'], analysis['optimized_tat']]
        ax1.bar(['현재 TAT', '최적화 후 TAT'], tat_comparison, color=['lightcoral', 'lightgreen'])
        ax1.set_title('TAT 비교')
        ax1.set_ylabel('평균 TAT (분)')
        for i, v in enumerate(tat_comparison):
            ax1.text(i, v, f'{v:.1f}', ha='center', va='bottom')

        # 블록 할당 비교
        current_counts = analysis['block_counts']['current']
        optimized_counts = analysis['block_counts']['optimized']
        
        all_cargo_types = sorted(set(list(current_counts.keys()) + list(optimized_counts.keys())))
        current_values = [current_counts.get(ct, 0) for ct in all_cargo_types]
        optimized_values = [optimized_counts.get(ct, 0) for ct in all_cargo_types]
        
        x = np.arange(len(all_cargo_types))
        width = 0.35
        
        ax2.bar(x - width/2, current_values, width, label='현재', color='lightcoral')
        ax2.bar(x + width/2, optimized_values, width, label='최적화', color='lightgreen')
        ax2.set_title('블록 할당 비교')
        ax2.set_xlabel('화물 유형')
        ax2.set_ylabel('블록 수')
        ax2.set_xticks(x)
        ax2.set_xticklabels(all_cargo_types, rotation=45)
        ax2.legend()

        # 개선율 게이지 차트
                # 개선율 게이지 차트 부분을 다음과 같이 수정
        improvement = analysis['improvement']
        
        # 텍스트 기반 개선율 표시로 변경
        ax3.text(0.5, 0.5, f'TAT 개선율\n{abs(improvement):.1f}%', 
                ha='center', va='center', fontsize=14)
        if improvement > 0:
            ax3.text(0.5, 0.3, '개선됨', ha='center', va='center', 
                    color='green', fontsize=12)
        elif improvement < 0:
            ax3.text(0.5, 0.3, '악화됨', ha='center', va='center', 
                    color='red', fontsize=12)
        else:
            ax3.text(0.5, 0.3, '변화없음', ha='center', va='center', 
                    color='gray', fontsize=12)
        ax3.axis('off')
        

        plt.tight_layout()
        plt.show()  # 첫 번째 그래프 세트 표시

        # 시간대별 TAT 분포
        plt.figure(figsize=(15, 5))
        
        self.data['hour'] = pd.to_datetime(self.data['IN_DATE']).dt.hour
        self.data['TAT'] = (pd.to_datetime(self.data['OUT_DATE']) - 
                           pd.to_datetime(self.data['IN_DATE'])).dt.total_seconds() / 60
        
        hourly_tat = self.data.groupby('hour')['TAT'].mean()
        
        plt.plot(hourly_tat.index, hourly_tat.values, marker='o', color='blue')
        plt.title('시간대별 평균 TAT')
        plt.xlabel('시간')
        plt.ylabel('평균 TAT (분)')
        plt.grid(True, alpha=0.3)
        
        for i, v in hourly_tat.items():
            plt.text(i, v, f'{v:.1f}', ha='center', va='bottom')
            
        plt.tight_layout()
        plt.show()  # 두 번째 그래프 표시

        # 화물 유형별 TAT 분포
        plt.figure(figsize=(15, 5))
        sns.boxplot(data=self.data, x='CARGO_TYPE', y='TAT')
        plt.title('화물 유형별 TAT 분포')
        plt.xlabel('화물 유형')
        plt.ylabel('TAT (분)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()  # 세 번째 그래프 표시
        
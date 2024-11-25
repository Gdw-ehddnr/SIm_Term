import pandas as pd
import numpy as np
from typing import Dict, List
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numba
import matplotlib.font_manager as fm

class BlockOptimizer:
    CONTAINER_PER_BLOCK = 1739  # 블록당 최대 컨테이너 수

    def __init__(self, data: pd.DataFrame, sample_size: int = 10000):
        """초기화"""
        if len(data) > sample_size:
            self.data = data.sample(n=sample_size, random_state=42)
        else:
            self.data = data.copy()  # 원본 데이터 복사
            
        # 데이터 전처리
        self.data['IN_DATE'] = pd.to_datetime(self.data['IN_DATE'])
        self.data['OUT_DATE'] = pd.to_datetime(self.data['OUT_DATE'])
        self.data['TAT'] = (self.data['OUT_DATE'] - self.data['IN_DATE']).dt.total_seconds() / 60
        
        # CARGO_TYPE을 카테고리 타입에서 일반 문자열로 변환
        self.data['CARGO_TYPE'] = self.data['CARGO_TYPE'].astype(str)
        
        # T 타입 통합 (DG, DR, RF -> T)
        self.data.loc[self.data['CARGO_TYPE'].isin(['DG', 'DR', 'RF']), 'CARGO_TYPE'] = 'T'
        
        # 전체 unique 블록 리스트 생성 (원본 데이터에서)
        self.current_blocks = sorted(data['BLOCK'].unique())
        
        # 화물 타입별 물동량 계산 (BN, DO, ED 제외)
        self.cargo_volumes = self.data['CARGO_TYPE'].value_counts()
        self.cargo_types = [ct for ct in self.cargo_volumes.index 
                        if ct not in ['BN', 'DO', 'ED']]
        
        print(f"총 블록 수: {len(self.current_blocks)}")
        print(f"시뮬레이션 데이터 크기: {len(self.data):,}개 레코드")
        print(f"화물 타입별 물동량: {dict(self.cargo_volumes)}")

    def calculate_block_stats(self) -> Dict:
        """각 블록별 TAT 평균 및 표준편차 계산"""
        block_stats = {}
        
        for block in self.current_blocks:
            block_data = self.data[self.data['BLOCK'] == block]['TAT']
            if len(block_data) > 0:
                block_stats[block] = {
                    'mean': block_data.mean(),
                    'std': block_data.std(),
                    'count': len(block_data)
                }
        
        return block_stats

    def simulate_block_ratio(self, num_simulations: int = 100) -> Dict:
        """블록 비율 시뮬레이션"""
        best_allocation = None
        best_tat = float('inf')
        results = []
        
        # 전체 블록 수 정의
        total_blocks = len(self.current_blocks)
        
        # 현재 블록 할당 상태 파악
        current_block_counts = {
            cargo_type: len(self.data[self.data['CARGO_TYPE'] == cargo_type]['BLOCK'].unique())
            for cargo_type in self.cargo_types
        }
        
        print(f"전체 unique 블록 수: {total_blocks}")
        print("현재 화물 타입별 사용 블록 수 (중복 포함):", current_block_counts)
        
        # 블록별 TAT 통계 계산
        self.block_stats = self.calculate_block_stats()
        
        for sim_num in range(num_simulations):
            try:
                # 1. 전체 블록 풀 준비
                all_blocks = self.current_blocks.copy()
                block_allocation = {}
                
                # 2. T 타입 할당 (2~4개)
                t_volume = len(self.data[self.data['CARGO_TYPE'] == 'T'])
                if t_volume > 0:
                    t_count = min(4, max(2, sum(current_block_counts.get(t, 0) for t in ['DG', 'DR', 'RF'])))
                else:
                    t_count = 2  # 물동량이 없어도 최소 2개
                
                t_blocks = list(np.random.choice(all_blocks, size=t_count, replace=False))
                block_allocation['T'] = t_blocks
                all_blocks = [b for b in all_blocks if b not in t_blocks]
                
                # 3. 나머지 타입 처리
                remaining_blocks = len(all_blocks)
                optimizable_types = ['MT', 'GP', 'AK']
                
                # 4. 물동량 계산
                type_volumes = {
                    ct: len(self.data[self.data['CARGO_TYPE'] == ct])
                    for ct in optimizable_types
                }
                total_volume = sum(type_volumes.values())
                
                # 5. 블록 수 계산
                block_counts = {}
                remaining_count = remaining_blocks
                
                for i, cargo_type in enumerate(optimizable_types):
                    if i == len(optimizable_types) - 1:
                        # 마지막 타입에는 남은 블록 모두 할당
                        block_counts[cargo_type] = remaining_count
                    else:
                        # 물동량 비율에 따라 할당
                        ratio = type_volumes[cargo_type] / total_volume if total_volume > 0 else 1/len(optimizable_types)
                        count = max(1, min(int(remaining_blocks * ratio), remaining_count - (len(optimizable_types) - i - 1)))
                        block_counts[cargo_type] = count
                        remaining_count -= count
                
                # 6. 실제 블록 할당
                for cargo_type, count in block_counts.items():
                    if count > 0:
                        selected_blocks = list(np.random.choice(all_blocks, size=count, replace=False))
                        block_allocation[cargo_type] = selected_blocks
                        all_blocks = [b for b in all_blocks if b not in selected_blocks]
                
                # 7. 할당 검증
                total_allocated = sum(len(blocks) for blocks in block_allocation.values())
                if total_allocated != total_blocks:
                    print(f"Warning: 블록 할당 불일치 ({total_allocated} != {total_blocks})")
                    continue
                
                # 8. TAT 계산
                current_tat = self.calculate_tat(block_allocation)
                
                if current_tat is not None and not np.isnan(current_tat):
                    if current_tat < best_tat:
                        best_tat = current_tat
                        best_allocation = block_allocation.copy()
                    
                    results.append({
                        'allocation': block_allocation.copy(),
                        'tat': current_tat,
                        'block_counts': {k: len(v) for k, v in block_allocation.items()}
                    })
                    print(f"시뮬레이션 {sim_num+1} 성공: TAT = {current_tat:.2f}")
            
            except Exception as e:
                print(f"시뮬레이션 {sim_num+1} 실행 중 오류 발생: {e}")
                continue
        
        if not results:
            raise ValueError("유효한 시뮬레이션 결과가 없습니다.")
        
        print("\n=== 최적화 결과 ===")
        print(f"최적 TAT: {best_tat:.2f}")
        print("블록 할당:", {k: len(v) for k, v in best_allocation.items()})
        
        return {
            'best_allocation': best_allocation,
            'best_tat': best_tat,
            'all_results': sorted(results, key=lambda x: x['tat'])
        }
    def calculate_tat(self, block_allocation: Dict[str, List[str]]) -> float:
        """정규분포 기반 TAT 계산"""
        if not block_allocation:
            return None
            
        total_tat = 0.0
        total_count = 0
        
        try:
            for cargo_type, blocks in block_allocation.items():
                if not blocks:  # 빈 블록 리스트 체크
                    continue
                    
                # T 타입 처리
                if cargo_type == 'T':
                    cargo_data = self.data[self.data['CARGO_TYPE'].isin(['DG', 'DR', 'RF'])]
                else:
                    cargo_data = self.data[self.data['CARGO_TYPE'] == cargo_type]
                
                if len(cargo_data) == 0:
                    continue
                
                for block in blocks:
                    block_stats = self.block_stats.get(block)
                    if not block_stats:
                        continue
                        
                    # 블록당 컨테이너 수 제한
                    container_count = min(
                        len(cargo_data),
                        self.CONTAINER_PER_BLOCK
                    )
                    
                    if container_count > 0:
                        # 정규분포에서 TAT 샘플링
                        tat_samples = np.random.normal(
                            loc=block_stats['mean'],
                            scale=max(1, block_stats['std']),  # 표준편차가 0이 되는 것 방지
                            size=container_count
                        )
                        
                        # 음수 TAT 방지
                        valid_tats = np.maximum(0, tat_samples)
                        
                        total_tat += np.sum(valid_tats)
                        total_count += container_count
            
            if total_count == 0:
                return None
                
            average_tat = total_tat / total_count
            
            # nan 체크
            if np.isnan(average_tat):
                return None
                
            return average_tat
            
        except Exception as e:
            print(f"TAT 계산 중 오류 발생: {str(e)}")
            return None
    
    def assign_blocks_to_layout(self, block_counts: Dict[str, int]) -> Dict[str, List[str]]:
        """각 화물 타입별로 실제 블록 할당"""
        if not block_counts:
            return None
        
        try:
            # 가용 블록 목록
            available_blocks = self.current_blocks.copy()
            block_allocation = {}
            
            # 1. T 타입 먼저 할당 (DG, DR, RF 통합)
            if 'T' in block_counts and block_counts['T'] > 0:
                if len(available_blocks) < block_counts['T']:
                    print(f"Warning: T 타입에 대한 가용 블록 부족")
                    return None
                    
                t_blocks = list(np.random.choice(
                    available_blocks, 
                    size=block_counts['T'],
                    replace=False
                ))
                block_allocation['T'] = t_blocks
                available_blocks = [b for b in available_blocks if b not in t_blocks]
            
            # 2. 나머지 타입 할당 (MT, GP, AK)
            other_types = [ct for ct in block_counts.keys() if ct != 'T']
            for cargo_type in other_types:
                if block_counts[cargo_type] > 0:
                    if len(available_blocks) < block_counts[cargo_type]:
                        print(f"Warning: {cargo_type}에 대한 가용 블록 부족")
                        return None
                        
                    selected_blocks = list(np.random.choice(
                        available_blocks,
                        size=block_counts[cargo_type],
                        replace=False
                    ))
                    block_allocation[cargo_type] = selected_blocks
                    available_blocks = [b for b in available_blocks if b not in selected_blocks]
            
            # 3. 할당 검증
            total_allocated = sum(len(blocks) for blocks in block_allocation.values())
            total_required = sum(block_counts.values())
            
            if total_allocated != total_required:
                print(f"Warning: 블록 할당 불일치 ({total_allocated} != {total_required})")
                return None
                
            return block_allocation
            
        except Exception as e:
            print(f"블록 할당 중 오류 발생: {str(e)}")
            return None

    def visualize_results(self, simulation_results: Dict):
        """시뮬레이션 결과 시각화"""
        plt.rcParams['font.family'] = 'Malgun Gothic'  
        plt.style.use('default')
        font_list = [font.name for font in fm.fontManager.ttflist]
        print("사용 가능한 폰트 목록:", font_list)
    
        # 한글 폰트 설정 (여러 옵션 시도)
        for font_name in ['Malgun Gothic', '맑은 고딕', 'NanumGothic', '나눔고딕']:
            if font_name in font_list:
                plt.rcParams['font.family'] = font_name
                print(f"선택된 폰트: {font_name}")
                break
        else:
            print("Warning: 적절한 한글 폰트를 찾을 수 없습니다.")
        
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
        
        # 1. 현재와 최적화된 블록 할당 비교 (히트맵)
        plt.figure(figsize=(20, 8))
        
        # 현재 블록 할당 상태
        current_allocation = np.zeros((4, len(self.current_blocks)))  # 4: ['MT', 'GP', 'AK', 'T']
        cargo_types = ['MT', 'GP', 'AK', 'T']
        
        for i, block in enumerate(sorted(self.current_blocks)):
            block_types = self.data[self.data['BLOCK'] == block]['CARGO_TYPE'].unique()
            for j, cargo_type in enumerate(cargo_types):
                if cargo_type in block_types:
                    current_allocation[j, i] = 1
        
        # 최적화된 블록 할당 상태
        optimal_allocation = np.zeros((4, len(self.current_blocks)))
        for cargo_type, blocks in simulation_results['best_allocation'].items():
            type_idx = cargo_types.index(cargo_type)
            for block in blocks:
                block_idx = sorted(self.current_blocks).index(block)
                optimal_allocation[type_idx, block_idx] = 1
        
        # 두 개의 서브플롯 생성
        plt.subplot(2, 1, 1)
        plt.imshow(current_allocation, cmap='YlOrRd', aspect='auto')
        plt.title('현재 블록별 화물 타입 할당')
        plt.yticks(range(4), cargo_types)
        plt.xticks(range(len(self.current_blocks)), sorted(self.current_blocks), rotation=45)
        plt.colorbar()
        
        plt.subplot(2, 1, 2)
        plt.imshow(optimal_allocation, cmap='YlOrRd', aspect='auto')
        plt.title('최적화된 블록별 화물 타입 할당')
        plt.yticks(range(4), cargo_types)
        plt.xticks(range(len(self.current_blocks)), sorted(self.current_blocks), rotation=45)
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()
        
        # 2. TAT 개선 효과 시각화
        current_tat = self.calculate_current_tat()
        optimal_tat = simulation_results['best_tat']
        
        plt.figure(figsize=(10, 6))
        tat_data = [current_tat, optimal_tat]
        labels = ['현재 TAT', '최적화 TAT']
        colors = ['skyblue', 'lightcoral']
        
        bars = plt.bar(labels, tat_data, color=colors)
        plt.title('TAT 개선 효과 비교')
        plt.ylabel('평균 TAT (분)')
        
        # 개선율 표시
        improvement = ((current_tat - optimal_tat) / current_tat) * 100
        plt.text(0, current_tat, f'{current_tat:.1f}분', ha='center', va='bottom')
        plt.text(1, optimal_tat, f'{optimal_tat:.1f}분\n({improvement:.1f}% 개선)', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.show()

    def calculate_current_tat(self) -> float:
        """현재 상태의 TAT 계산"""
        total_tat = 0
        total_count = 0
        
        for block in self.current_blocks:
            block_data = self.data[self.data['BLOCK'] == block]
            if len(block_data) > 0:
                total_tat += block_data['TAT'].mean() * len(block_data)
                total_count += len(block_data)
        
        return total_tat / total_count if total_count > 0 else 0
from pathlib import Path
from src.utils.data_loader import PortDataLoader
from src.optimization.block_optimizer import BlockOptimizer  # Add this import
import seaborn as sns
import time

def load_data():
    """데이터 로드"""
    data_dir = Path(r"C:\Users\ddd26\OneDrive\바탕 화면\부경대학교\3학년 2학기\sim_term\port_simulation\data\input")
    loader = PortDataLoader(data_dir)
    loader.load_data()
    return loader

def analyze_blocks():
    """블록 분석 수행"""
    loader = load_data()
    results = loader.analyze_block_distribution()
    loader.print_analysis_results(results)

def visualize_blocks():
    """블록 분포 시각화"""
    loader = load_data()
    loader.visualize_block_distribution()

def optimize_blocks():
    """블록 비율 최적화 시뮬레이션 실행"""
    loader = load_data()
    
    # 샘플 크기 입력 받기
    while True:
        try:
            sample_size = input("시뮬레이션할 데이터 샘플 크기를 입력하세요 (기본값: 10000): ")
            if not sample_size:  # 입력없이 엔터를 누른 경우
                sample_size = 10000
            else:
                sample_size = int(sample_size)
            break
        except ValueError:
            print("올바른 숫자를 입력하세요.")
    
    print(f"\n{sample_size:,}개의 데이터로 시뮬레이션을 시작합니다...")
    optimizer = BlockOptimizer(loader.data, sample_size=sample_size)
    
    # 시뮬레이션 반복 횟수 입력 받기
    while True:
        try:
            num_simulations = input("시뮬레이션 반복 횟수를 입력하세요 (기본값: 100): ")
            if not num_simulations:
                num_simulations = 100
            else:
                num_simulations = int(num_simulations)
            break
        except ValueError:
            print("올바른 숫자를 입력하세요.")
    
    print(f"\n{num_simulations}회 시뮬레이션을 시작합니다...")
    start_time = time.time()
    simulation_results = optimizer.simulate_block_ratio(num_simulations=num_simulations)
    end_time = time.time()
    
    analysis = optimizer.analyze_results(simulation_results)
    
    print(f"\n=== 시뮬레이션 결과 (소요시간: {end_time - start_time:.1f}초) ===")
    print(f"현재 평균 TAT: {analysis['current_tat']:.2f} 분")
    print(f"최적화 후 평균 TAT: {analysis['optimized_tat']:.2f} 분")
    print(f"개선율: {analysis['improvement']:.2f}%")
    
    print("\n현재 블록 할당:")
    for cargo_type, count in analysis['block_counts']['current'].items():
        print(f"{cargo_type}: {count}개 블록")
    
    print("\n최적화된 블록 할당:")
    for cargo_type, count in analysis['block_counts']['optimized'].items():
        print(f"{cargo_type}: {count}개 블록")

    print("\n시뮬레이션 결과를 시각화합니다...")
    optimizer.visualize_results(analysis)

def main():
    while True:
        print("\n=== 항만 데이터 분석 시스템 ===")
        print("1. 데이터 로드")
        print("2. 블록별 분석")
        print("3. 블록 분포 시각화")
        print("4. 블록 비율 최적화 시뮬레이션")
        print("0. 종료")
        
        try:
            choice = input("\n원하는 작업을 선택하세요: ")
            
            if choice == "1":
                loader = load_data()
                print("데이터 로드 완료")
            elif choice == "2":
                analyze_blocks()
            elif choice == "3":
                visualize_blocks()
            elif choice == "4":
                optimize_blocks()
            elif choice == "0":
                print("프로그램을 종료합니다.")
                break
            else:
                print("잘못된 선택입니다. 다시 선택해주세요.")
                
        except Exception as e:
            print(f"에러 발생: {str(e)}")
            import traceback
            print(traceback.format_exc())

if __name__ == "__main__":
    main()
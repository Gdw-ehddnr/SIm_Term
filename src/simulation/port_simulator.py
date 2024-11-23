import simpy

class PortSimulator:
    def __init__(self, env, block_config):
        self.env = env
        self.blocks = self.initialize_blocks(block_config)
        
    def container_arrival(self, container_type):
        """컨테이너 도착 프로세스"""
        while True:
            # 컨테이너 생성 및 처리
            yield self.env.timeout(self.get_arrival_time())
            
    def process_container(self, container):
        """컨테이너 처리 프로세스"""
        # 블록 할당 및 처리
        pass
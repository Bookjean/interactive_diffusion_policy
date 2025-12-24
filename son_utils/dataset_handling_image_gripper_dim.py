import h5py
import numpy as np

file_path = '/home/ws/Downloads/RB10_dataset/pick_and_place/idp_data.hdf5'

with h5py.File(file_path, 'r+') as f:
    if 'data' in f:
        data_group = f['data']
        print(f"총 {len(data_group)}개의 demo 그룹을 검사합니다...")
        
        count = 0
        
        for demo_key in data_group.keys():
            # obs 그룹 접근 (예: data/demo_0/obs)
            demo_group = data_group[demo_key]
            if 'obs' in demo_group:
                obs_group = demo_group['obs']
                
                if 'gripper' in obs_group:
                    # 1. 기존 데이터 읽기
                    gripper_data = obs_group['gripper'][:]
                    
                    # 2. 차원 확인 및 변환 (N,) -> (N, 1)
                    # ndim이 1인 경우에만 작업 수행
                    if gripper_data.ndim == 1:
                        # (N, 1)로 형태 변경
                        new_data = gripper_data.reshape(-1, 1)
                        
                        # 3. 안전한 교체 과정
                        # (1) 임시 데이터셋 생성
                        obs_group.create_dataset('gripper_tmp', data=new_data)
                        
                        # (2) 기존 데이터셋 삭제
                        del obs_group['gripper']
                        
                        # (3) 임시 데이터셋 이름을 원래 이름으로 변경
                        obs_group.move('gripper_tmp', 'gripper')
                        
                        count += 1
                    
                    elif gripper_data.ndim == 2 and gripper_data.shape[1] == 1:
                         # 이미 (N, 1)인 경우 패스
                         pass
                    else:
                        print(f"주의 [{demo_key}]: 예상치 못한 shape입니다. {gripper_data.shape}")

        print("-" * 30)
        print(f"완료: 총 {count}개의 데모에서 'gripper' 차원을 (N, 1)로 변환했습니다.")
        
        # 결과 확인 (첫 번째 데모)
        first_demo = list(data_group.keys())[0]
        print(f"확인 [{first_demo}]: gripper shape = {data_group[first_demo]['obs']['gripper'].shape}")
        
    else:
        print("오류: 'data' 그룹을 찾을 수 없습니다.")
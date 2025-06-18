from pathlib import Path


tasks = [
    ('h5_data/lift_barrier.h5', 2),
]
# tasks = [
#     ('h5_data/lift_barrier_pointcloud.h5', 2),
# ]

for path, agent_num in tasks:
    path = Path(path)
    dataset_path = str(path)

    output_path = str(path.parent.parent / path.name)
    # cmd = f'python script/image/extract.py --dataset_path={dataset_path} --output_path={output_path} --load_num 50 --agent_num {agent_num}'
    cmd = f'python script/pointcloud/extract.py --dataset_path={dataset_path} --output_path={output_path} --load_num 50 --agent_num {agent_num}'
    
    print(cmd)
    # os.system(cmd)
    # break
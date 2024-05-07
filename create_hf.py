# # # # # # # from datasets import Dataset, DatasetDict
# # # # # # # import numpy as np
# # # # # # # import pandas as pd
# # # # # # # import os
# # # # # # # import glob
# # # # # # # from PIL import Image

# # # # # # # def load_image(path):
# # # # # # #     return np.array(Image.open(path))

# # # # # # # def process_trajectory(directory):
# # # # # # #     color_images = sorted(glob.glob(os.path.join(directory, 'color_image_*.jpg')))
# # # # # # #     depth_images = sorted(glob.glob(os.path.join(directory, 'depth_image_*.npy')))
# # # # # # #     poses = sorted(glob.glob(os.path.join(directory, 'pose_*.npy')))
    
# # # # # # #     return {
# # # # # # #         'color_images': [Image.open(img) for img in color_images],
# # # # # # #         'depth_images': [np.load(img) for img in depth_images],
# # # # # # #         'poses': [np.load(pose) for pose in poses]
# # # # # # #     }

# # # # # # # def create_dataset(data_dir):
# # # # # # #     trajectories = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
# # # # # # #     data = [process_trajectory(traj) for traj in trajectories]
    
# # # # # # #     dataset = Dataset.from_pandas(pd.DataFrame(data))
# # # # # # #     return dataset

# # # # # # # data_dir = './stream_data'
# # # # # # # dataset = create_dataset(data_dir)
# # # # # # # dataset.save_to_disk('./saved')

# # # # # # from datasets import Dataset, DatasetDict
# # # # # # import numpy as np
# # # # # # import os
# # # # # # import glob
# # # # # # from PIL import Image

# # # # # # def process_trajectory(directory):
# # # # # #     color_images = sorted(glob.glob(os.path.join(directory, 'color_image_*.jpg')))
# # # # # #     depth_images = sorted(glob.glob(os.path.join(directory, 'depth_image_*.npy')))
# # # # # #     poses = sorted(glob.glob(os.path.join(directory, 'pose_*.npy')))
    
# # # # # #     return {
# # # # # #         'color_image_paths': color_images,
# # # # # #         'depth_image_paths': depth_images,
# # # # # #         'pose_paths': poses
# # # # # #     }

# # # # # # def create_dataset(data_dir):
# # # # # #     trajectories = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
# # # # # #     data = [process_trajectory(traj) for traj in trajectories]
    
# # # # # #     dataset = Dataset.from_dict(data)
# # # # # #     return dataset

# # # # # # data_dir = './stream_data'
# # # # # # dataset = create_dataset(data_dir)
# # # # # # dataset.save_to_disk('./saved_dataset')

# # # # # from datasets import Dataset
# # # # # import os
# # # # # import glob

# # # # # def process_trajectory(directory):
# # # # #     color_images = sorted(glob.glob(os.path.join(directory, 'color_image_*.jpg')))
# # # # #     depth_images = sorted(glob.glob(os.path.join(directory, 'depth_image_*.npy')))
# # # # #     poses = sorted(glob.glob(os.path.join(directory, 'pose_*.npy')))
    
# # # # #     return {
# # # # #         'color_image_paths': color_images,
# # # # #         'depth_image_paths': depth_images,
# # # # #         'pose_paths': poses
# # # # #     }

# # # # # def create_dataset(data_dir):
# # # # #     trajectories = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
# # # # #     # Initialize empty lists to collect all data
# # # # #     data = {
# # # # #         'color_image_paths': [],
# # # # #         'depth_image_paths': [],
# # # # #         'pose_paths': []
# # # # #     }

# # # # #     # Aggregate data across all trajectories
# # # # #     for traj in trajectories:
# # # # #         traj_data = process_trajectory(traj)
# # # # #         for key in data:
# # # # #             data[key].extend(traj_data[key])
    
# # # # #     # Convert collected data into a HuggingFace Dataset
# # # # #     dataset = Dataset.from_dict(data)
# # # # #     return dataset

# # # # # data_dir = './stream_data'
# # # # # dataset = create_dataset(data_dir)
# # # # # dataset.save_to_disk('./saved_dataset')

# # # # from datasets import Dataset
# # # # import os
# # # # import glob

# # # # def process_trajectory(directory):
# # # #     color_images = sorted(glob.glob(os.path.join(directory, 'color_image_*.jpg')))
# # # #     depth_images = sorted(glob.glob(os.path.join(directory, 'depth_image_*.npy')))
# # # #     poses = sorted(glob.glob(os.path.join(directory, 'pose_*.npy')))
    
# # # #     return {
# # # #         'color_image_paths': color_images,
# # # #         'depth_image_paths': depth_images,
# # # #         'pose_paths': poses
# # # #     }

# # # # def save_trajectory_datasets(data_dir, save_base_dir):
# # # #     trajectories = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
# # # #     for i, traj in enumerate(trajectories):
# # # #         traj_data = process_trajectory(traj)
# # # #         # Create a dataset for each trajectory
# # # #         dataset = Dataset.from_dict(traj_data)
# # # #         # Create a unique directory for each dataset
# # # #         save_dir = os.path.join(save_base_dir, f'{i + 1}')
# # # #         os.makedirs(save_dir, exist_ok=True)
# # # #         # Save the dataset to its directory
# # # #         dataset.save_to_disk(save_dir)

# # # # data_dir = './stream_data'
# # # # save_base_dir = './saved_datasets'
# # # # save_trajectory_datasets(data_dir, save_base_dir)

# # # from datasets import Dataset
# # # import numpy as np
# # # import os
# # # import glob
# # # from PIL import Image

# # # def process_trajectory(directory):
# # #     color_images = sorted(glob.glob(os.path.join(directory, 'color_image_*.jpg')))
# # #     depth_images = sorted(glob.glob(os.path.join(directory, 'depth_image_*.npy')))
# # #     poses = sorted(glob.glob(os.path.join(directory, 'pose_*.npy')))
    
# # #     observations = [{'color_image': Image.open(img_path), 'depth_image': np.load(depth_path)}
# # #                     for img_path, depth_path in zip(color_images, depth_images)]
# # #     actions = [np.load(pose) for pose in poses]  # Assuming poses are actions
# # #     rewards = [0] * len(actions)  # Placeholder for rewards, adjust as necessary
# # #     timesteps = list(range(len(actions)))  # Assuming consecutive timesteps
    
# # #     return {
# # #         'observations': observations,
# # #         'actions': actions,
# # #         'rewards': rewards,
# # #         'timesteps': timesteps
# # #     }

# # # def save_trajectory_datasets(data_dir, save_base_dir):
# # #     trajectories = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
# # #     for i, traj in enumerate(trajectories):
# # #         traj_data = process_trajectory(traj)
# # #         dataset = Dataset.from_dict(traj_data)
# # #         save_dir = os.path.join(save_base_dir, f'trajectory_{i}')
# # #         os.makedirs(save_dir, exist_ok=True)
# # #         dataset.save_to_disk(save_dir)

# # # data_dir = './stream_data'
# # # save_base_dir = './saved_datasets'
# # # save_trajectory_datasets(data_dir, save_base_dir)

# # import tensorflow as tf
# # import os
# # import glob
# # import numpy as np

# # def create_step_dataset(color_image_paths, depth_image_paths, pose_paths):
# #     step_data = {
# #         'is_first': [],
# #         'is_last': [],
# #         'observation': [],  # Use color image paths as observation
# #         'action': [],       # Pose data will be used as action
# #         'reward': [],       # Dummy rewards, modify if actual data is available
# #         'is_terminal': [],
# #         'discount': []      # Dummy discounts, modify if actual data is available
# #     }

# #     num_steps = len(color_image_paths)
# #     for i in range(num_steps):
# #         step_data['is_first'].append(i == 0)
# #         step_data['is_last'].append(i == num_steps - 1)
# #         step_data['observation'].append(color_image_paths[i])
# #         step_data['action'].append(np.load(pose_paths[i]))  # Load pose data as action
# #         step_data['reward'].append(0)  # Assume no rewards, adjust as necessary
# #         step_data['is_terminal'].append(False)
# #         step_data['discount'].append(0.99)  # Example discount factor

# #     return tf.data.Dataset.from_tensor_slices(step_data)

# # def process_trajectory(directory, episode_id):
# #     color_images = sorted(glob.glob(os.path.join(directory, 'color_image_*.jpg')))
# #     depth_images = sorted(glob.glob(os.path.join(directory, 'depth_image_*.npy')))
# #     poses = sorted(glob.glob(os.path.join(directory, 'pose_*.npy')))
    
# #     steps_dataset = create_step_dataset(color_images, depth_images, poses)
    
# #     episode_metadata = {
# #         'episode_id': episode_id,
# #         'agent_id': 'agent_1',  # Example agent ID
# #         'environment_config': 'config_1',  # Example environment configuration
# #         'experiment_id': 'experiment_1',  # Example experiment ID
# #         'invalid': False
# #     }
    
# #     return {
# #         'steps': steps_dataset,
# #         'metadata': episode_metadata
# #     }

# # def create_rlds_dataset(data_dir):
# #     trajectories = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
# #     episodes = []
    
# #     for i, traj in enumerate(trajectories):
# #         episode = process_trajectory(traj, f'episode_{i}')
# #         episodes.append(episode)
    
# #     return episodes

# # def save_rlds_dataset(episodes, save_base_dir):
# #     os.makedirs(save_base_dir, exist_ok=True)
    
# #     for i, episode in enumerate(episodes):
# #         episode_path = os.path.join(save_base_dir, f'episode_{i}')
# #         os.makedirs(episode_path, exist_ok=True)
# #         tf.data.experimental.save(episode['steps'], os.path.join(episode_path, 'steps'))
# #         with open(os.path.join(episode_path, 'metadata.json'), 'w') as f:
# #             import json
# #             json.dump(episode['metadata'], f)

# # data_dir = './stream_data'
# # save_base_dir = './saved_rlds_datasets'
# # episodes = create_rlds_dataset(data_dir)
# # save_rlds_dataset(episodes, save_base_dir)

# import tensorflow as tf
# import numpy as np
# import os
# import glob
# import json

# def serialize_example(color_image_path, pose_data):
#     """
#     Serialize data to TFRecord format.
#     """
#     image_string = open(color_image_path, 'rb').read()
#     feature = {
#         'is_first': tf.train.Feature(int64_list=tf.train.Int64List(value=[int('first' in color_image_path)])),
#         'is_last': tf.train.Feature(int64_list=tf.train.Int64List(value=[int('last' in color_image_path)])),
#         'is_terminal': tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
#         'action': tf.train.Feature(float_list=tf.train.FloatList(value=pose_data.tolist())),
#         'observation': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
#         'reward': tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
#         'discount': tf.train.Feature(float_list=tf.train.FloatList(value=[1.0])),
#     }
#     example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
#     return example_proto.SerializeToString()

# def process_trajectory(directory):
#     color_images = sorted(glob.glob(os.path.join(directory, 'color_image_*.jpg')))
#     poses = [np.load(f) for f in sorted(glob.glob(os.path.join(directory, 'pose_*.npy')))]
#     return color_images, poses

# def write_tfrecord(data_dir, output_dir):
#     """
#     Write all the data to a TFRecord file.
#     """
#     color_images, poses = process_trajectory(data_dir)
#     tfrecord_filename = os.path.join(output_dir, 'dataset-train.tfrecord-00000-of-00001')
#     with tf.io.TFRecordWriter(tfrecord_filename) as writer:
#         for color_image, pose in zip(color_images, poses):
#             serialized_example = serialize_example(color_image, pose)
#             writer.write(serialized_example)

# def create_metadata_files(output_dir):
#     dataset_info = {
#         "citation": "// TODO: Add your citation",
#         "description": "Description of the dataset in Markdown.",
#         "fileFormat": "tfrecord",
#         "moduleName": "custom_dataset_module.custom_dataset",
#         "name": "custom_dataset",
#         "releaseNotes": {
#             "1.0.0": "Initial release."
#         },
#         "splits": [
#             {
#                 "filepathTemplate": "dataset-train.tfrecord-{SHARD_INDEX}-of-{NUM_SHARDS}",
#                 "name": "train",
#                 "numBytes": "123456789",  # Update this with the actual size
#                 "shardLengths": ["100"]  # Update with actual counts
#             }
#         ],
#         "version": "1.0.0"
#     }

#     with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
#         json.dump(dataset_info, f)

# def main():
#     data_dir = './stream_data'
#     output_dir = './output'
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Assuming a single trajectory for simplicity. Loop or modify for multiple trajectories.
#     write_tfrecord(data_dir, output_dir)
#     create_metadata_files(output_dir)

# if __name__ == '__main__':
#     main()

# # # # #

import os
import json
from datasets import Dataset, Image
import huggingface_hub
import numpy as np

# Login to Hugging Face (ensure your credentials are securely managed)
huggingface_hub.login('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')

base_path = 'stream_data'
trajectories = os.listdir(base_path)

# Containers for the dataset
all_rgb = []
all_depth = []
all_poses = []
all_meta = []
all_trajectory_ids = []

# Process each trajectory
for traj in trajectories:
    traj_path = os.path.join(base_path, traj)
    
    # RGB images
    rgb_files = [os.path.join(traj_path, f) for f in os.listdir(traj_path) if 'color_image' in f and f.endswith('.jpg')]
    all_rgb.extend(rgb_files)
    all_trajectory_ids.extend([traj] * len(rgb_files))

    # Depth images
    depth_files = [os.path.join(traj_path, f) for f in os.listdir(traj_path) if 'depth_image' in f and f.endswith('.npy')]
    all_depth.extend(depth_files)
    if len(depth_files) < len(rgb_files):  # Adding placeholder numpy arrays if fewer depth files
        additional_depth = [np.zeros((480, 640), dtype=np.uint16) for _ in range(len(rgb_files) - len(depth_files))]
        all_depth.extend(additional_depth)

    # Poses
    pose_files = [os.path.join(traj_path, f) for f in os.listdir(traj_path) if 'pose' in f and f.endswith('.npy')]
    all_poses.extend(pose_files)
    if len(pose_files) < len(rgb_files):  # Adding placeholder numpy arrays if fewer pose files
        additional_poses = [np.zeros((7,)) for _ in range(len(rgb_files) - len(pose_files))]
        all_poses.extend(additional_poses)

    # Metadata
    json_files = [os.path.join(traj_path, f) for f in os.listdir(traj_path) if f.endswith('.json')]
    if json_files:
        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)
                all_meta.append(data)
    else:
        # Ensure at least one dummy entry if no json files
        all_meta.extend([{'dummy_field': None}] * len(rgb_files))

# Load npy files
depth_data = [np.load(f) if isinstance(f, str) else f for f in all_depth]
pose_data = [np.load(f) if isinstance(f, str) else f for f in all_poses]

# Create dataset
ds = Dataset.from_dict({
    'trajectory_id': all_trajectory_ids,
    'rgb': all_rgb,
    'depth': depth_data,
    'poses': pose_data,
    'meta': all_meta
}).cast_column('rgb', Image())

# # Show dataset structure and preview data
# print("Dataset Structure:")
# print(ds)
# print("\nPreview of the first few rows:")
# print(ds[:5])

# Optionally push to Hugging Face Hub (uncomment to use)
ds.push_to_hub('CarterKruse/non-prehensile-manipulation')





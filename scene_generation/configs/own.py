from configs.base import get_config as base_get_config
# 用于colmap格式的GS的场景格式转换

def get_config():
  config = base_get_config()
  config.format = 'colmap'
  config.up_axis = 'z'
  config.decimation_factor = 4
  config.depth_max = 5.0
  config.slice_distance = 4.0
  config.slice_overlap = 3.0
  config.buffer_distance = 0.75
  config.to_ig_euler_xyz = (0.0, 0.0, 0.0)
  config.slice_direction = '-'
  config.load_mesh = True
  config.voxel_size = 0.005
  config.min_poses_per_segment = 4
  return config

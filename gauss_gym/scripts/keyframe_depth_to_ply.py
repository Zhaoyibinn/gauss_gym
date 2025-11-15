"""Convert Polycam-style keyframe depth maps and poses into a merged point cloud."""
import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import imageio.v3 as iio
import numpy as np

OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


@dataclass
class FrameRecord:
  image_name: str
  c2w: np.ndarray
  intrinsics: dict


def _load_camera_matrix(camera_json: Path) -> Tuple[np.ndarray, dict]:
  """Load a single camera JSON file and return its extrinsic matrix and intrinsics.

  The keyframe camera files store the 3x4 matrix as ``t_ij`` entries with a row
  permutation relative to the ``transforms.json`` export. We reorder the rows so
  that the matrix matches the Nerfstudio-style camera-to-world transform.
  """
  with camera_json.open('r', encoding='utf-8') as f:
    data = json.load(f)

  # Reconstruct the 4x4 camera-to-world matrix. The row order in the keyframe
  # metadata follows (1, 2, 0) relative to transforms.json, so we permute here.
  c2w = np.eye(4, dtype=np.float32)
  c2w[0, :4] = [data['t_20'], data['t_21'], data['t_22'], data['t_23']]
  c2w[1, :4] = [data['t_00'], data['t_01'], data['t_02'], data['t_03']]
  c2w[2, :4] = [data['t_10'], data['t_11'], data['t_12'], data['t_13']]

  intrinsics = {
    'fx': float(data['fx']),
    'fy': float(data['fy']),
    'cx': float(data['cx']),
    'cy': float(data['cy']),
    'width': int(data['width']),
    'height': int(data['height']),
  }

  return c2w, intrinsics


def _depth_to_camera_points(
  depth: np.ndarray,
  intrinsics: dict,
  depth_scale: float,
  confidence: Optional[np.ndarray] = None,
  confidence_threshold: int = 0,
  stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
  """Project a depth map into 3D camera coordinates.

  Args:
      depth: Raw depth image (``uint16`` or ``float``) with shape ``(H, W)``.
      intrinsics: Dictionary with ``fx, fy, cx, cy, width, height`` describing
        the original full-resolution camera parameters.
      depth_scale: Divisor applied to raw depth to obtain metres.
      confidence: Optional confidence map aligned with the depth image.
      confidence_threshold: Minimum confidence value to keep a depth sample.

  Returns:
      Tuple containing:
        - Array of 3D points in the camera frame with shape ``(N, 3)``.
        - Pixel column indices ``u`` (N,).
        - Pixel row indices ``v`` (N,).
        - Horizontal scaling factor ``sx`` between depth and original resolution.
        - Vertical scaling factor ``sy`` between depth and original resolution.
  """
  depth = depth.astype(np.float32) / depth_scale

  base_mask = depth > 0
  if confidence is not None:
    base_mask &= confidence >= confidence_threshold

  mask = base_mask
  if stride > 1:
    sampled_mask = np.zeros_like(base_mask)
    sampled_mask[::stride, ::stride] = True
    mask &= sampled_mask

  if not np.any(mask):
    empty = np.empty((0,), dtype=np.int64)
    return np.empty((0, 3), dtype=np.float32), empty, empty, 1.0, 1.0

  v_coords, u_coords = np.nonzero(mask)
  z = depth[mask]

  h_depth, w_depth = depth.shape
  sx = w_depth / intrinsics['width']
  sy = h_depth / intrinsics['height']

  fx = intrinsics['fx'] * sx
  fy = intrinsics['fy'] * sy
  cx = intrinsics['cx'] * sx
  cy = intrinsics['cy'] * sy

  x = (u_coords - cx) * z / fx
  y = (v_coords - cy) * z / fy

  points = np.stack([x, y, z], axis=-1).astype(np.float32)
  return points, u_coords.astype(np.int64), v_coords.astype(np.int64), sx, sy


def _camera_to_world(points_cam: np.ndarray, c2w: np.ndarray) -> np.ndarray:
  """Transform camera-frame points into world coordinates."""
  if points_cam.size == 0:
    return points_cam
  points_cam_gl = points_cam @ OPENCV_TO_OPENGL.T
  rot = c2w[:3, :3]
  trans = c2w[:3, 3]
  return (points_cam_gl @ rot.T) + trans


def _rotmat_to_quaternion(rot: np.ndarray) -> np.ndarray:
  """Convert a rotation matrix to a normalized quaternion (w, x, y, z)."""
  m00, m01, m02 = rot[0]
  m10, m11, m12 = rot[1]
  m20, m21, m22 = rot[2]
  trace = m00 + m11 + m22
  if trace > 0.0:
    s = np.sqrt(trace + 1.0) * 2.0
    qw = 0.25 * s
    qx = (m21 - m12) / s
    qy = (m02 - m20) / s
    qz = (m10 - m01) / s
  elif (m00 > m11) and (m00 > m22):
    s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
    qw = (m21 - m12) / s
    qx = 0.25 * s
    qy = (m01 + m10) / s
    qz = (m02 + m20) / s
  elif m11 > m22:
    s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
    qw = (m02 - m20) / s
    qx = (m01 + m10) / s
    qy = 0.25 * s
    qz = (m12 + m21) / s
  else:
    s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
    qw = (m10 - m01) / s
    qx = (m02 + m20) / s
    qy = (m12 + m21) / s
    qz = 0.25 * s
  quat = np.array([qw, qx, qy, qz], dtype=np.float64)
  quat /= np.linalg.norm(quat)
  return quat


def _c2w_to_colmap_pose(c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Convert camera-to-world matrix (OpenGL) to COLMAP quaternion and translation."""
  rot_c2w = c2w[:3, :3].astype(np.float64)
  trans_c2w = c2w[:3, 3].astype(np.float64)
  rot_w2c_gl = rot_c2w.T
  trans_w2c_gl = -rot_w2c_gl @ trans_c2w
  rot_w2c_cv = OPENCV_TO_OPENGL.astype(np.float64) @ rot_w2c_gl
  trans_w2c_cv = OPENCV_TO_OPENGL.astype(np.float64) @ trans_w2c_gl
  quat = _rotmat_to_quaternion(rot_w2c_cv)
  return quat, trans_w2c_cv


def _write_colmap_text_model(records: List[FrameRecord], output_dir: Path) -> None:
  """Write COLMAP text files (cameras, images, points3D) from frame records."""
  if not records:
    raise RuntimeError('No frames available to export COLMAP model.')

  output_dir.mkdir(parents=True, exist_ok=True)

  reference_intrinsics = records[0].intrinsics
  width = reference_intrinsics['width']
  height = reference_intrinsics['height']
  fx = reference_intrinsics['fx']
  fy = reference_intrinsics['fy']
  cx = reference_intrinsics['cx']
  cy = reference_intrinsics['cy']

  for record in records[1:]:
    intr = record.intrinsics
    float_keys = ('fx', 'fy', 'cx', 'cy')
    int_keys = ('width', 'height')
    # if any(abs(float(intr[key]) - float(reference_intrinsics[key])) > 1e-5 for key in float_keys):
    #   raise ValueError('Inconsistent intrinsics encountered while exporting COLMAP model.')
    if any(int(intr[key]) != int(reference_intrinsics[key]) for key in int_keys):
      raise ValueError('Inconsistent image sizes encountered while exporting COLMAP model.')

  cameras_txt = output_dir / 'cameras.txt'
  with cameras_txt.open('w', encoding='utf-8') as f:
    f.write('# Camera list with one line of data per camera:\n')
    f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
    f.write('# Number of cameras: 1\n')
    f.write(f'1 PINHOLE {width} {height} {fx:.10f} {fy:.10f} {cx:.10f} {cy:.10f}\n\n')

  images_txt = output_dir / 'images.txt'
  with images_txt.open('w', encoding='utf-8') as f:
    f.write('# Image list with two lines of data per image:\n')
    f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
    f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
    f.write(f'# Number of images: {len(records)}, mean observations per image: 0\n')
    for idx, record in enumerate(records, start=1):
      quat, trans = _c2w_to_colmap_pose(record.c2w)
      quat_str = ' '.join(f'{v:.10f}' for v in quat)
      trans_str = ' '.join(f'{v:.10f}' for v in trans)
      f.write(f'{idx} {quat_str} {trans_str} 1 {record.image_name}\n\n')

  points_txt = output_dir / 'points3D.txt'
  with points_txt.open('w', encoding='utf-8') as f:
    f.write('# 3D point list with one line of data per point:\n')
    f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
    f.write('# Number of points: 0, mean track length: 0\n\n')

def save_point_cloud(points: np.ndarray, output_path: Path, colors: Optional[np.ndarray] = None) -> None:
  """Write an ASCII PLY point cloud with optional vertex colors."""
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open('w', encoding='ascii') as f:
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write(f'element vertex {len(points)}\n')
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    if colors is not None and len(colors) == len(points):
      f.write('property uchar red\n')
      f.write('property uchar green\n')
      f.write('property uchar blue\n')
    f.write('end_header\n')
    if colors is not None and len(colors) == len(points):
      colors_u8 = colors.astype(np.uint8)
      data = np.column_stack((points, colors_u8))
      fmt = ['%.6f', '%.6f', '%.6f', '%d', '%d', '%d']
      np.savetxt(f, data, fmt=fmt)
    else:
      np.savetxt(f, points, fmt='%.6f')

def _copy_images(src_dir: Path, dst_dir: Path) -> None:
  """Copy RGB images for COLMAP consumption."""
  if not src_dir.exists():
    raise FileNotFoundError(f'Image source directory not found: {src_dir}')

  dst_dir.mkdir(parents=True, exist_ok=True)
  for image_file in sorted(src_dir.glob('*')):
    if image_file.is_file():
      shutil.copy2(image_file, dst_dir / image_file.name)


def run(
  root_dir: Path,
  depth_scale: float,
  confidence_threshold: int,
  stride: int,
  export_colmap: bool = True,
  copy_images: bool = True,
) -> None:
  root_dir = root_dir.expanduser().resolve()
  keyframe_dir = root_dir / 'keyframes'
  if not keyframe_dir.exists():
    raise FileNotFoundError(f'Keyframe directory not found under root: {keyframe_dir}')

  cameras_dir = keyframe_dir / 'cameras'
  depth_dir = keyframe_dir / 'depth'
  confidence_dir = keyframe_dir / 'confidence'
  images_dir = keyframe_dir / 'images'

  colmap_dir: Optional[Path] = (root_dir / 'sparse' / '0') if export_colmap else None
  images_output_dir: Optional[Path] = (root_dir / 'images') if copy_images else None

  if not cameras_dir.exists() or not depth_dir.exists():
    raise FileNotFoundError('Expected "cameras" and "depth" directories under the keyframe path.')

  if images_output_dir is not None:
    if images_dir.exists():
      _copy_images(images_dir, images_output_dir)
    else:
      print('Warning: keyframe images directory missing, skipping image copy.')

  points_world = []
  colors_world: List[np.ndarray] = []
  frame_records: List[FrameRecord] = []
  depth_files = sorted(depth_dir.glob('*.png'))
  if not depth_files:
    raise FileNotFoundError('No depth PNGs found under the keyframe directory.')

  for depth_path in depth_files:
    stem = depth_path.stem
    camera_path = cameras_dir / f'{stem}.json'
    if not camera_path.exists():
      continue

    c2w, intrinsics = _load_camera_matrix(camera_path)
    depth_img = iio.imread(depth_path)

    image_name = None
    if images_dir.exists():
      image_candidate = images_dir / f'{stem}.jpg'
      if image_candidate.exists():
        image_name = image_candidate.name
      else:
        image_candidate = images_dir / f'{stem}.png'
        if image_candidate.exists():
          image_name = image_candidate.name

    if colmap_dir is not None:
      if image_name is not None:
        frame_records.append(
          FrameRecord(
            image_name=image_name,
            c2w=c2w.copy(),
            intrinsics=intrinsics.copy(),
          )
        )
      else:
        print(f'Warning: no image found for stem {stem}, skipping COLMAP entry.')

    confidence_img = None
    if confidence_dir.exists():
      conf_path = confidence_dir / f'{stem}.png'
      if conf_path.exists():
        confidence_img = iio.imread(conf_path)

    points_cam, u_coords, v_coords, sx, sy = _depth_to_camera_points(
      depth_img,
      intrinsics,
      depth_scale=depth_scale,
      confidence=confidence_img,
      confidence_threshold=confidence_threshold,
      stride=stride,
    )

    if points_cam.size == 0:
      continue

    colors = None
    if image_name is not None:
      color_path = images_dir / image_name
      if color_path.exists():
        color_img = iio.imread(color_path)
        if color_img.ndim == 2:
          color_img = np.stack([color_img] * 3, axis=-1)
        if color_img.shape[-1] > 3:
          color_img = color_img[..., :3]
        if not np.issubdtype(color_img.dtype, np.uint8):
          if np.issubdtype(color_img.dtype, np.floating):
            scale = 255.0 if color_img.max() <= 1.0 else 1.0
            color_img = np.clip(color_img * scale, 0.0, 255.0).astype(np.uint8)
          else:
            color_img = np.clip(color_img, 0, 255).astype(np.uint8)

        u_orig = np.clip(np.round(u_coords / sx).astype(int), 0, color_img.shape[1] - 1)
        v_orig = np.clip(np.round(v_coords / sy).astype(int), 0, color_img.shape[0] - 1)
        colors = color_img[v_orig, u_orig]
      else:
        print(f'Warning: image file missing for stem {stem}, using black colors.')

    if colors is None:
      colors = np.zeros((points_cam.shape[0], 3), dtype=np.uint8)

    points_world.append(_camera_to_world(points_cam, c2w))
    colors_world.append(colors.astype(np.uint8))

  if not points_world:
    raise RuntimeError('No valid points were generated from the provided keyframes.')

  merged_points = np.concatenate(points_world, axis=0)
  merged_colors = np.concatenate(colors_world, axis=0) if colors_world else None
  # save_point_cloud(merged_points, output_path, merged_colors)

  if colmap_dir is not None:
    if not frame_records:
      raise RuntimeError('COLMAP export requested but no valid frame records were collected.')
    colmap_dir.mkdir(parents=True, exist_ok=True)
    save_point_cloud(merged_points, colmap_dir / 'points3D.ply', merged_colors)
    _write_colmap_text_model(frame_records, colmap_dir)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Fuse keyframe depth maps into a point cloud.')
  parser.add_argument(
    '--root-dir',
    type=Path,
    default=Path('dataset/zyb_home_colmap'),
    help='Root directory containing keyframes/images/sparse subfolders.',
  )
  parser.add_argument(
    '--depth-scale',
    type=float,
    default=1000.0,
    help='Scale factor to convert raw depth units to metres (value divided by this factor).',
  )
  parser.add_argument(
    '--confidence-threshold',
    type=int,
    default=0,
    help='Drop depth samples with confidence below this threshold.',
  )
  parser.add_argument(
    '--stride',
    type=int,
    default=5,
    help='Use every Nth pixel in each dimension when projecting depth.',
  )
  parser.add_argument(
    '--no-colmap',
    action='store_true',
    help='Skip generating COLMAP outputs under root_dir/sparse/0.',
  )
  parser.add_argument(
    '--no-copy-images',
    action='store_true',
    help='Skip copying keyframe RGB images into root_dir/images.',
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  run(
    root_dir=args.root_dir,
    depth_scale=args.depth_scale,
    confidence_threshold=args.confidence_threshold,
    stride=args.stride,
    export_colmap=not args.no_colmap,
    copy_images=not args.no_copy_images,
  )


if __name__ == '__main__':
  main()

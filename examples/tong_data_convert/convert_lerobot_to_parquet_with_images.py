import os
# Use standard tqdm for loop, not notebook version unless required
from tqdm import tqdm
import numpy as np
import torch # Needed again for image tensor processing
import datasets
# Import Image feature type
from datasets import Features, Value, Sequence, load_dataset, Image as HFImage
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import logging
import shutil
import gc
import json # Needed for metadata files
from huggingface_hub import hf_hub_download, HfApi # For copying metadata
import math
import cv2 # Import OpenCV for resizing if needed

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)
logging.getLogger("lerobot").setLevel(logging.INFO)
logging.getLogger("huggingface_hub").setLevel(logging.INFO)

# --- Configuration ---
repo_id = "Loki0929/so100_duck_v20"
# Reflecting that images are now included
output_dir_structured = f"./{repo_id.replace('/', '_')}_full_parquet_with_images"
WRITER_BATCH_SIZE = 500 # Reduce batch size slightly, images are larger
CHUNK_SIZE = 500 # Episodes per output chunk directory

# --- Create Output Directories ---
output_data_dir = os.path.join(output_dir_structured, "data")
output_meta_dir = os.path.join(output_dir_structured, "meta")

logging.info(f"Cleaning up existing output directory: {output_dir_structured}")
if os.path.exists(output_dir_structured):
    shutil.rmtree(output_dir_structured)

os.makedirs(output_dir_structured)
os.makedirs(output_data_dir)
os.makedirs(output_meta_dir)
logging.info(f"Created directory structure under: {output_dir_structured}")

# --- Get Source Metadata & Determine Dimensions ---
logging.info(f"Fetching metadata for {repo_id}...")
try:
    metadata = LeRobotDatasetMetadata(repo_id)
    logging.info("Source metadata fetched.")
except Exception as e:
     logging.exception(f"Failed to fetch metadata: {e}")
     raise

camera_keys = metadata.camera_keys
fps = metadata.fps
action_dim = metadata.features['action']['shape'][0]
state_dim = metadata.features['observation.state']['shape'][0]

# Determine image dimensions (H, W, C) from metadata
h, w, c = 0, 0, 0
if camera_keys:
    first_cam_key = camera_keys[0]
    # Check based on the structure observed in the target info.json example
    info_json_style_key = f"observation.images.{first_cam_key.split('.')[-1]}"
    target_example_key_found = False
    if info_json_style_key in metadata.features:
         meta_shape = metadata.features[info_json_style_key].get('shape')
         if meta_shape and len(meta_shape) == 3:
             h, w, c = meta_shape
             target_example_key_found = True
             logging.info(f"Found image shape {h,w,c} from metadata key '{info_json_style_key}'.")

    # Fallback: Check original camera key if the info.json style wasn't found or didn't work
    if not target_example_key_found and first_cam_key in metadata.features:
        meta_shape = metadata.features[first_cam_key].get('shape')
        if meta_shape and len(meta_shape) == 3:
             h, w, c = meta_shape
             logging.info(f"Found image shape {h,w,c} from fallback metadata key '{first_cam_key}'.")
        # Check dtype as well if shape exists
        elif 'dtype' in metadata.features[first_cam_key] and metadata.features[first_cam_key]['dtype'] == 'video':
             meta_shape = metadata.features[first_cam_key].get('shape')
             if meta_shape and len(meta_shape) == 3:
                  h, w, c = meta_shape
                  logging.info(f"Found video shape {h,w,c} from key '{first_cam_key}'.")


# If still not found, use the default from the target info.json
if h == 0 or w == 0 or c == 0:
    h, w, c = 480, 640, 3 # Default from target info.json
    logging.warning(f"Could not determine image dimensions reliably from metadata. Using default shape: H={h}, W={w}, C={c}")
else:
    logging.info(f"Using image shape: H={h}, W={w}, C={c}")


# --- Define Target Features (Including Images using HFImage) ---
new_features_dict = {
    # Non-image features
    'action': Sequence(Value('float32'), length=action_dim),
    'observation.state': Sequence(Value('float32'), length=state_dim),
    'timestamp': Value('float32'),
    'frame_index': Value('int64'),  # Index within the episode
    'episode_index': Value('int64'), # Index of the episode itself
    'index': Value('int64'),        # Global frame index across all episodes
    'task_index': Value('int64'),   # Task index (assuming 0)
}
# Add image features using datasets.Image
for key in camera_keys:
    new_features_dict[key] = HFImage() # Let datasets handle encoding/decoding

new_features = Features(new_features_dict)
logging.info("\nTarget dataset features (for Parquet files, including Images) defined:")
logging.info(new_features)


# --- Determine All Episode IDs using LeRobotDataset (same as before) ---
logging.info("Determining available episodes by initializing LeRobotDataset...")
all_episode_ids = []
dummy_dataset_for_episodes = None
try:
    dummy_dataset_for_episodes = LeRobotDataset(repo_id)
    if hasattr(dummy_dataset_for_episodes, 'episode_indices'):
         indices = dummy_dataset_for_episodes.episode_indices
         all_episode_ids = indices.tolist() if isinstance(indices, np.ndarray) else list(indices)
         logging.info(f"Successfully retrieved episode indices via LeRobotDataset.episode_indices")
    elif hasattr(dummy_dataset_for_episodes, 'num_episodes') and dummy_dataset_for_episodes.num_episodes > 0:
         logging.warning("Using fallback: assuming sequential episodes 0 to num_episodes-1.")
         all_episode_ids = list(range(dummy_dataset_for_episodes.num_episodes))
    else:
        raise AttributeError("Cannot determine episode list.")

    if not all_episode_ids:
         raise ValueError("LeRobotDataset reported 0 episodes. Cannot proceed.")

    all_episode_ids.sort()
    logging.info(f"Found {len(all_episode_ids)} episodes. First few: {all_episode_ids[:10]}...")
    num_total_episodes_found = len(all_episode_ids)

except Exception as e:
    logging.exception(f"Failed to determine episode list using LeRobotDataset: {e}")
    raise
finally:
     if dummy_dataset_for_episodes is not None:
          if hasattr(dummy_dataset_for_episodes, 'close'):
               try: dummy_dataset_for_episodes.close()
               except Exception as close_e: logging.warning(f"Error closing dummy dataset: {close_e}")
          del dummy_dataset_for_episodes
          gc.collect()
          logging.info("Cleaned up dummy LeRobotDataset instance.")


# --- Define the generator function (Now includes image processing for HFImage) ---
def generate_episode_data(repo_id, episode_idx_to_load, start_global_index,
                          target_h, target_w, target_c, # Pass target dimensions
                          action_dim, state_dim, camera_keys): # Pass other needed info
    """
    Generator function to load and process data (including images) for a single episode.
    Yields dictionaries matching the `new_features` definition (with HFImage).
    """
    logging.debug(f"[Generator] Starting for Ep {episode_idx_to_load}, Start Global Idx: {start_global_index}")
    local_dataset = None
    num_frames = 0
    total_frames_yielded = 0
    # Create placeholder image once if needed
    placeholder_img_np = np.zeros((target_h, target_w, target_c), dtype=np.uint8)

    try:
        logging.debug(f"[Generator] Initializing LeRobotDataset for episode {episode_idx_to_load}")
        # Load only the specified single episode
        # Important: LeRobotDataset with transforms=None might still apply some defaults.
        # If images aren't float C,H,W [0,1], adjust processing below.
        local_dataset = LeRobotDataset(repo_id, episodes=[episode_idx_to_load])
        logging.debug(f"[Generator] LeRobotDataset initialized for episode {episode_idx_to_load}.")

        num_frames = local_dataset.num_frames if hasattr(local_dataset, 'num_frames') else 0
        if num_frames <= 0:
            logging.warning(f"[Generator] Ep {episode_idx_to_load}: num_frames={num_frames}. Skipping.")
        else:
            logging.info(f"[Generator] Ep {episode_idx_to_load}: Found {num_frames} frames.")

    except Exception as e:
         logging.exception(f"[Generator] Setup failed for episode {episode_idx_to_load}: {e}")
         num_frames = 0

    # Iterate through frames
    if local_dataset and num_frames > 0:
        for frame_idx_in_episode in range(num_frames):
            try:
                original_item = local_dataset[frame_idx_in_episode]

                # Essential keys check
                if 'frame_index' not in original_item or 'timestamp' not in original_item:
                    logging.warning(f"[Generator] Ep {episode_idx_to_load}, Rel Idx {frame_idx_in_episode}: Missing frame_index/timestamp. Skipping.")
                    continue

                # --- Extract Non-Image Data (Same as before) ---
                original_frame_index = original_item['frame_index']
                if hasattr(original_frame_index, 'item'): original_frame_index = original_frame_index.item()
                original_frame_index = int(original_frame_index)

                current_global_index = start_global_index + frame_idx_in_episode

                new_item = {
                    'frame_index': np.int64(original_frame_index),
                    'episode_index': np.int64(episode_idx_to_load),
                    'index': np.int64(current_global_index),
                    'task_index': np.int64(0),
                }

                ts = original_item.get('timestamp')
                if hasattr(ts, 'item'): ts = ts.item()
                new_item['timestamp'] = np.float32(ts if ts is not None else 0.0)

                state_data = original_item.get('observation.state')
                if hasattr(state_data, 'tolist'): state_list = state_data.squeeze().tolist()
                elif isinstance(state_data, (list, np.ndarray)): state_list = list(np.array(state_data).astype(np.float32).squeeze())
                else: state_list = ([0.0] * state_dim)
                new_item['observation.state'] = [np.float32(x) for x in state_list]

                action_data = original_item.get('action')
                if hasattr(action_data, 'tolist'): action_list = action_data.squeeze().tolist()
                elif isinstance(action_data, (list, np.ndarray)): action_list = list(np.array(action_data).astype(np.float32).squeeze())
                else: action_list = ([0.0] * action_dim)
                new_item['action'] = [np.float32(x) for x in action_list]


                # --- Process Image Data ---
                for key in camera_keys:
                    image_tensor = original_item.get(key) # Expect Float32 [0, 1], C,H,W from LeRobotDataset

                    # Validate tensor
                    if not torch.is_tensor(image_tensor) or image_tensor.numel() == 0 or len(image_tensor.shape) != 3:
                        logging.warning(f"[Generator] Ep {episode_idx_to_load}, Frame {original_frame_index}, Key '{key}': Invalid/missing image tensor. Using placeholder.")
                        new_item[key] = placeholder_img_np # Use placeholder NumPy array
                        continue

                    try:
                        # Clamp values just in case
                        image_tensor = torch.clamp(image_tensor, 0.0, 1.0)

                        # Convert C,H,W [0,1] float -> H,W,C [0,255] uint8 NumPy
                        # Ensure it's on CPU before numpy()
                        image_np_uint8 = (image_tensor.permute(1, 2, 0) * 255.0).byte().cpu().numpy()

                        # --- Resize if necessary ---
                        current_h, current_w, current_c = image_np_uint8.shape
                        if current_h != target_h or current_w != target_w or current_c != target_c:
                            logging.warning(f"[Generator] Ep {episode_idx_to_load}, Frame {original_frame_index}, Key '{key}': Shape mismatch ({current_h, current_w, current_c}) vs target ({target_h, target_w, target_c}). Resizing.")
                            # Ensure contiguous array for cv2
                            image_np_uint8_contiguous = np.ascontiguousarray(image_np_uint8)
                            resized_img = cv2.resize(image_np_uint8_contiguous, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                            # Handle potential channel changes during resize (e.g., grayscale input)
                            if len(resized_img.shape) == 2 and target_c == 1:
                                image_np_uint8 = np.expand_dims(resized_img, axis=-1) # Add channel dim
                            elif len(resized_img.shape) == 2 and target_c > 1:
                                logging.error(f"[Generator] Ep {episode_idx_to_load}, Frame {original_frame_index}, Key '{key}': Resize resulted in 2D array but target has {target_c} channels. Using placeholder.")
                                image_np_uint8 = placeholder_img_np
                            elif resized_img.shape == (target_h, target_w, target_c):
                                image_np_uint8 = resized_img
                            else:
                                logging.error(f"[Generator] Ep {episode_idx_to_load}, Frame {original_frame_index}, Key '{key}': Resize resulted in unexpected shape {resized_img.shape}. Using placeholder.")
                                image_np_uint8 = placeholder_img_np

                        # Ensure dtype is uint8
                        if image_np_uint8.dtype != np.uint8:
                            image_np_uint8 = image_np_uint8.astype(np.uint8)

                        # Assign the NumPy array. HFImage feature will handle encoding.
                        new_item[key] = image_np_uint8

                    except Exception as img_proc_e:
                        logging.exception(f"[Generator] Ep {episode_idx_to_load}, Frame {original_frame_index}, Key '{key}': Error processing image: {img_proc_e}. Using placeholder.")
                        new_item[key] = placeholder_img_np

                yield new_item
                total_frames_yielded += 1

            except IndexError:
                 logging.error(f"[Generator] Ep {episode_idx_to_load}: IndexError at rel idx {frame_idx_in_episode}. Stopping generation.")
                 break
            except Exception as frame_e:
                 frame_ref = f"original frame {original_item.get('frame_index', 'N/A')}" if 'original_item' in locals() else f"rel idx {frame_idx_in_episode}"
                 logging.warning(f"[Generator] Ep {episode_idx_to_load}, {frame_ref}: Error processing frame: {type(frame_e).__name__}. Skipping. Details: {frame_e}")
                 continue

    # Cleanup
    if local_dataset is not None:
        if hasattr(local_dataset, 'close'):
             try: local_dataset.close()
             except Exception as close_e: logging.warning(f"[Generator] Error closing dataset ep {episode_idx_to_load}: {close_e}")
        del local_dataset
    gc.collect()
    logging.info(f"[Generator] Finished Ep {episode_idx_to_load}. Yielded: {total_frames_yielded} frames.")


# --- Process and Save Each Episode (Including Images) ---
processed_episodes_info = []
skipped_episodes = []
global_frame_counter = 0
total_frames_processed = 0

logging.info(f"\n--- Starting Processing for {len(all_episode_ids)} Episodes ---")
for episode_idx in tqdm(all_episode_ids, desc="Processing Episodes"):
    chunk_index = episode_idx // CHUNK_SIZE
    chunk_dir_name = f"chunk-{chunk_index:03d}"
    chunk_output_dir = os.path.join(output_data_dir, chunk_dir_name)
    os.makedirs(chunk_output_dir, exist_ok=True)
    episode_filename = f"episode_{episode_idx:06d}.parquet"
    episode_output_path = os.path.join(chunk_output_dir, episode_filename)

    episode_dataset = None
    num_frames_in_episode = 0
    try:
        gen_kwargs = {
            "repo_id": repo_id,
            "episode_idx_to_load": episode_idx,
            "start_global_index": global_frame_counter,
            "target_h": h, "target_w": w, "target_c": c, # Pass target dims
            "action_dim": action_dim,
            "state_dim": state_dim,
            "camera_keys": camera_keys, # Pass camera keys
        }
        # Create dataset for the single episode
        episode_dataset = datasets.Dataset.from_generator(
            generate_episode_data,
            features=new_features, # Features now include HFImage()
            gen_kwargs=gen_kwargs,
            # Keep cache relatively small per generator run if memory is tight
            # cache_dir=f"./gen_cache_ep_{episode_idx}"
        )
        num_frames_in_episode = len(episode_dataset)

        if num_frames_in_episode == 0:
            logging.warning(f"Episode {episode_idx}: Generated dataset is empty. Skipping save.")
            skipped_episodes.append(episode_idx)
        else:
            if os.path.exists(episode_output_path):
                os.remove(episode_output_path)

            # Save to Parquet. HFImage feature handles image encoding.
            episode_dataset.to_parquet(episode_output_path, batch_size=WRITER_BATCH_SIZE)

            # Store info & update counters
            processed_episodes_info.append({
                "episode_id": episode_idx,
                "num_steps": num_frames_in_episode,
                "chunk_id": chunk_index
            })
            global_frame_counter += num_frames_in_episode
            total_frames_processed += num_frames_in_episode
            logging.info(f"Episode {episode_idx}: Saved {num_frames_in_episode} frames to {episode_output_path}. Global count: {global_frame_counter}")

    except datasets.exceptions.DatasetGenerationError as dge:
         logging.error(f"Episode {episode_idx}: DatasetGenerationError: {dge}. Skipping.")
         skipped_episodes.append(episode_idx)
    except Exception as e:
        logging.exception(f"Episode {episode_idx}: Unhandled error: {e}. Skipping.")
        skipped_episodes.append(episode_idx)
    finally:
        # Explicit cleanup
        if episode_dataset is not None:
             try:
                 episode_dataset.cleanup_cache_files()
             except Exception as cache_clean_e:
                 logging.warning(f"Episode {episode_idx}: Error cleaning cache: {cache_clean_e}")
             del episode_dataset
        if episode_idx % 5 == 0: # More frequent GC with images
             gc.collect()


logging.info(f"\n--- Finished Processing All Episodes ---")
logging.info(f"Successfully processed: {len(processed_episodes_info)} episodes.")
logging.info(f"Total frames saved: {total_frames_processed}")
if skipped_episodes:
    logging.warning(f"Skipped {len(skipped_episodes)} episodes: {skipped_episodes}")

# --- Copy Metadata Files (episodes.jsonl, stats.json, tasks.jsonl) ---
logging.info("\n--- Copying Metadata Files from Source Repository ---")
# No HfApi instance needed here if only using hf_hub_download for this part
files_to_copy = ["episodes.jsonl", "stats.json", "tasks.jsonl"]
for filename in files_to_copy:
    source_path = f"meta/{filename}" # Assumes file is at the repo root
    target_path = os.path.join(output_meta_dir, filename)
    logging.info(f"Attempting to download '{filename}' from {repo_id}...")
    try:
        # Directly attempt the download.
        # hf_hub_download will raise an error if the file is not found.
        hf_hub_download(
            repo_id=repo_id,
            filename=source_path,
            local_dir=output_dir_structured,
            local_dir_use_symlinks=False, # Ensure actual copy
            repo_type="dataset"
            # cache_dir=None # Optional: disable caching for this download if needed
        )
        # Check if the file exists at the target location after download attempt
        if os.path.exists(target_path):
            logging.info(f"Successfully downloaded '{filename}' to {target_path}")
        else:
            # This case might occur if download fails silently or permissions issues,
            # though hf_hub_download usually raises an exception on failure.
            logging.error(f"Download command for '{filename}' completed without error, but file not found at {target_path}. Check permissions or download behavior.")

    except Exception as e:
        # Catch errors during download (e.g., file not found, network issues)
        logging.error(f"Failed to download '{filename}' from {repo_id}: {type(e).__name__} - {e}")
        # You could add more specific error checking here if desired, e.g.:
        # from huggingface_hub.utils import EntryNotFoundError
        # if isinstance(e, EntryNotFoundError):
        #    logging.warning(f"File '{filename}' does not exist in repo {repo_id}.")
        # else:
        #    logging.error(f"Failed to download '{filename}': {type(e).__name__} - {e}")


# --- Generate info.json (Structure should be mostly correct from before) ---
logging.info("\n--- Generating info.json ---")

source_features = metadata.features if hasattr(metadata, 'features') else {}

def get_feature_info(key, default_shape, default_dtype, default_names=None):
    feature = source_features.get(key, {})
    names = feature.get("names", default_names)
    # Ensure names list length matches shape if possible
    shape = feature.get("shape", default_shape)
    if isinstance(shape, list) and len(shape) == 1 and isinstance(names, list) and len(names) != shape[0]:
         logging.warning(f"Feature '{key}': Mismatch between names length ({len(names)}) and shape ({shape[0]}). Using default names.")
         names = default_names # Fallback or generate defaults
    elif names is None and isinstance(shape, list) and len(shape) == 1:
         # Generate default names if None
         if 'action' in key: names = [f"action_dim_{i}" for i in range(shape[0])]
         elif 'state' in key: names = [f"state_dim_{i}" for i in range(shape[0])]


    return {
        "dtype": feature.get("dtype", default_dtype),
        "shape": shape,
        "names": names
    }

info_features = {}
action_feat_info = get_feature_info('action', [action_dim], 'float32')
info_features['action'] = action_feat_info

state_feat_info = get_feature_info('observation.state', [state_dim], 'float32')
info_features['observation.state'] = state_feat_info

# Images - use determined h,w,c and camera_keys
for key in camera_keys:
    info_key = f"observation.images.{key.split('.')[-1]}"
    info_features[info_key] = {
        "dtype": "image", # Matches HFImage() usage
        "shape": [h, w, c], # Use dimensions determined earlier
        "names": ["height", "width", "channels"],
    }

# Other features - shape [1] as per target example info.json
info_features['timestamp'] = {"dtype": "float32", "shape": [1], "names": None}
info_features['frame_index'] = {"dtype": "int64", "shape": [1], "names": None}
info_features['episode_index'] = {"dtype": "int64", "shape": [1], "names": None}
info_features['index'] = {"dtype": "int64", "shape": [1], "names": None}
info_features['task_index'] = {"dtype": "int64", "shape": [1], "names": None}

num_processed_episodes = len(processed_episodes_info)
total_output_chunks = math.ceil(num_processed_episodes / CHUNK_SIZE) if num_processed_episodes > 0 else 0

info_data = {
    "codebase_version": "v2.0",
    "robot_type": "so100", # Assuming based on repo name/example
    "total_episodes": num_total_episodes_found,
    "total_frames": total_frames_processed,
    "total_tasks": 1,
    "total_videos": num_total_episodes_found * len(camera_keys), # Estimate
    "total_chunks": total_output_chunks,
    "chunks_size": 1000, # From example, meaning still ambiguous
    "fps": fps,
    "splits": {"train": f"0:{num_total_episodes_found}"},
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4", # Placeholder
    "features": info_features
}

info_path = os.path.join(output_meta_dir, "info.json")
try:
    with open(info_path, 'w') as f:
        json.dump(info_data, f, indent=4)
    logging.info(f"Saved generated: {info_path}")
except Exception as e:
    logging.error(f"Failed to save info.json: {e}")

# --- Optional: Verification (Check image loading) ---
if processed_episodes_info:
    # ... (find first episode path, same as before) ...
    first_processed_ep_info = sorted(processed_episodes_info, key=lambda x: x['episode_id'])[0]
    first_ep_idx = first_processed_ep_info["episode_id"]
    first_ep_chunk_idx = first_processed_ep_info["chunk_id"]
    first_ep_filename = f"episode_{first_ep_idx:06d}.parquet"
    first_ep_path = os.path.join(output_data_dir, f"chunk-{first_ep_chunk_idx:03d}", first_ep_filename)


    if os.path.exists(first_ep_path):
        logging.info(f"\n--- Verifying saved episode {first_ep_idx} (incl. images) ---")
        try:
            # Reload with the features including HFImage
            reloaded_dataset_ep = load_dataset(
                "parquet",
                data_files=first_ep_path,
                features=new_features, # Critical to use the correct features
                split="train[:2]" # Load first 2 rows
            )

            logging.info(f"Episode {first_ep_idx} reloaded from Parquet (first 2 rows):")
            logging.info(reloaded_dataset_ep)

            if len(reloaded_dataset_ep) > 0:
                first_item_reloaded = reloaded_dataset_ep[0]
                logging.info("\nFirst item from reloaded Parquet dataset:")
                logging.info(f"  Keys: {list(first_item_reloaded.keys())}")
                # ... (log non-image fields as before) ...
                logging.info(f"  Frame Index: {first_item_reloaded.get('frame_index')}")
                logging.info(f"  Global Index: {first_item_reloaded.get('index')}")

                # Check image fields
                for key in camera_keys:
                     if key in first_item_reloaded:
                          img_data = first_item_reloaded[key]
                          # HFImage decodes to PIL Image by default
                          logging.info(f"  Image '{key}' type: {type(img_data)}")
                          if hasattr(img_data, 'size'): # Check if it looks like a PIL Image
                              logging.info(f"    Image '{key}' size (W, H): {img_data.size}, mode: {img_data.mode}")
                              # Convert to numpy to check shape if needed
                              img_np = np.array(img_data)
                              logging.info(f"    Image '{key}' numpy shape (H, W, C): {img_np.shape}, dtype: {img_np.dtype}")
                          else:
                              logging.warning(f"    Image '{key}' reloaded data doesn't seem to be a PIL Image.")
                     else:
                         logging.warning(f"Camera key '{key}' not found in reloaded item.")
            else:
                 logging.info("Reloaded dataset slice is empty.")
        except Exception as e:
            logging.exception(f"Failed to reload or verify {first_ep_path}: {e}")
    else:
        logging.warning(f"Could not find file to verify: {first_ep_path}")
else:
    logging.warning("No episodes processed, skipping verification.")


# Final cleanup trigger
gc.collect()
logging.info(f"\nScript finished. Output with images in Parquet generated in: {output_dir_structured}")
logging.info("Metadata files (episodes.jsonl, stats.json, tasks.jsonl) were copied.")
logging.info("info.json was generated.")
logging.info("Parquet files in 'data/' now contain both non-image features and image features (using datasets.Image).")
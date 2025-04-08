from huggingface_hub import HfApi
import os
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/home/tong/lerobot/Loki0929_so100_duck_v20_full_parquet_with_images",
    repo_id="Loki0929/so100_duck_v20_full_parquet_with_images",
    repo_type="dataset",
)

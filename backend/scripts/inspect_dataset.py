from datasets import (
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset_builder,
)


def inspect_dataset(repo_id: str):
    print(f"Inspecting dataset: {repo_id}")
    try:
        configs = get_dataset_config_names(repo_id)
        print(f"Configurations found: {configs}")
        
        for config in configs:
            print(f"\n--- Config: {config} ---")
            try:
                splits = get_dataset_split_names(repo_id, config)
                print(f"Splits: {splits}")
                
                builder = load_dataset_builder(repo_id, config)
                print(f"Features: {builder.info.features}")
                
            except Exception as e:
                print(f"Error inspecting config {config}: {e}")
                
    except Exception as e:
        print(f"Error fetching configs: {e}")
        
        # Fallback for parquet structure if get_dataset_config_names fails
        print("\nAttempting to load builder directly (default config)...")
        try:
            builder = load_dataset_builder(repo_id)
            print(f"Features: {builder.info.features}")
            if builder.info.splits:
                for split_name, split_info in builder.info.splits.items():
                    print(f"Split: {split_name}, Num Examples: {split_info.num_examples}")
        except Exception as fallback_e:
            print(f"Fallback failed: {fallback_e}")

if __name__ == "__main__":
    inspect_dataset("ai4bharat/MSMARCO-XI")

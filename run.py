# run.py

from pipeline_controller import PipelineController
from config.load_config import load_config
from utils.seed import set_global_seed

def main():
    # 1. Load configuration
    config = load_config("config/config.yaml")

    # 2. Set reproducibility
    set_global_seed(config["seed"])

    # 3. Initialize pipeline
    pipeline = PipelineController(config)

    # 4. Run pipeline
    pipeline.run()

if __name__ == "__main__":
    main()

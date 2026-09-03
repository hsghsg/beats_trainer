from pathlib import Path

from beats_trainer import BEATsTrainer, Config


def main() -> None:
    """加载已训练好的 MIMII checkpoint，并单独执行测试集评估。"""
    config = Config()

    config.experiment_name = "mimii_pump_baseline"

    config.data.sample_rate = 16000
    config.data.batch_size = 16
    config.data.num_workers = 4

    config.model.freeze_backbone = True
    config.model.fine_tune_backbone = False

    config.training.max_epochs = 30
    config.training.learning_rate = 1e-4
    config.training.patience = 8
    config.training.gpus = 1

    checkpoint_path = Path(
        #r"logs\mimii_pump_baseline\version_1\checkpoints\mimii_pump_baseline-epoch=25-val_accuracy=0.943.ckpt"
        r"logs\mimii_pump_baseline\version_1\checkpoints\mimii_pump_baseline-epoch=27-val_accuracy=0.943.ckpt"
    )

    trainer = BEATsTrainer.from_split_directories(
        train_dir="data_ready/train",
        val_dir="data_ready/val",
        test_dir="data_ready/test",
        config=config,
    )

    test_results = trainer.trainer.test(
        trainer.model,
        datamodule=trainer.data_module,
        ckpt_path=str(checkpoint_path),
        weights_only=False,
    )

    print(test_results)


if __name__ == "__main__":
    main()
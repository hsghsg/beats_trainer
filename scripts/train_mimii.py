from beats_trainer import BEATsTrainer, Config


def main() -> None:
    """使用整理后的 MIMII 数据集训练 BEATs 异常声音分类模型。"""
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

    trainer = BEATsTrainer.from_split_directories(
        train_dir="data_ready/train",
        val_dir="data_ready/val",
        test_dir="data_ready/test",
        config=config,
    )

    #trainer.train()
    trainer.test()


if __name__ == "__main__":
    main()
from pathlib import Path

from Source.ConfigUtils import LoadYaml, ValidateFeatureConfig, ValidateProjectConfig


def test_project_config_loads():
    config = LoadYaml(Path("Configs/ProjectConfig.yaml"))
    ValidateProjectConfig(config)


def test_feature_configs_load():
    for config_name in ["ConfigA", "ConfigB", "ConfigC"]:
        config = LoadYaml(Path("Configs") / f"{config_name}.yaml")
        ValidateFeatureConfig(config, ExpectedName=config_name)

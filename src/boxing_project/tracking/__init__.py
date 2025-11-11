from pathlib import Path


# Всі модулі трекінгу мають спільний шлях до стандартної YAML-конфігурації.
# Використовуємо Path, щоб обчислити його відносно цього файлу незалежно від CWD.
DEFAULT_TRACKING_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "tracking.yaml"
)

__all__ = ["DEFAULT_TRACKING_CONFIG_PATH"]

"""exp.py 已被拆分到 provider.py、user.py 和 scenario.py。
保留此文件仅用于向后兼容，直接调用 scenario.create_example_scenario。
"""

from scenario import create_example_scenario


if __name__ == "__main__":
    create_example_scenario()

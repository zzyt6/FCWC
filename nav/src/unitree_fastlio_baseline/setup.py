from glob import glob
import os

from setuptools import find_packages, setup


package_name = "unitree_fastlio_baseline"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="unitree",
    maintainer_email="unitree@example.com",
    description="FAST-LIO Mid-360 indoor mapping baseline with a Viser web UI.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "viser_bridge = unitree_fastlio_baseline.viser_bridge:main",
        ],
    },
)

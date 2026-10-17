from glob import glob
import os

from setuptools import find_packages, setup


package_name = "unitree_nav2_web"


setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "behavior_trees"), glob("behavior_trees/*.xml")),
        (os.path.join("share", package_name, "web"), glob("web/*")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="unitree",
    maintainer_email="unitree@example.com",
    description="Canvas and HTTP/rosbridge orchestration for G1 Nav2.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "nav_web_bridge = unitree_nav2_web.nav_web_bridge:main",
            "cart_navigation = unitree_nav2_web.cart_navigation:main",
        ],
    },
)

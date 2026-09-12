from setuptools import find_packages, setup

package_name = "policy_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Anonymous",
    maintainer_email="anonymous@example.com",
    description=(
        "STeP-Cost runtime policy bridge for detour-gated semantic-motion "
        "TTL persistence in Navigation2."
    ),
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "policy_bridge = policy_bridge.policybridge:main",
        ],
    },
)

from setuptools import setup

package_name = "fr3_skill_server"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="lonnie",
    maintainer_email="lonnie@example.com",
    description="FR3 skill server + utilities",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "fr3_skill_server = fr3_skill_server.skill_server:main",
            "joint_state_filter = fr3_skill_server.joint_state_filter:main",
        ],
    },
)


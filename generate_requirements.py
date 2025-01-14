import platform
import re

def generate_requirements_txt(uv_lock_content: str, python_version: str, platform_system: str):
    """Generates a requirements.txt file from uv-requirements.txt content."""

    requirements = []
    package_pattern = re.compile(r"\[\[package\]\]\n"
                                 r"name = \"(?P<name>[^\"]+)\"\n"
                                 r"version = \"(?P<version>[^\"]+)\""
                                 r"(?P<dependencies>(?:.*marker = \"(?P<marker>[^\"]+)\")?(?:.*extra = \"(?P<extra>[^\"]+)\")?.*)*")

    blacklisted = [
        "pywin32" # windows only?,
    ]

    add_link = { 
        "torch-shampoo" : "git+https://github.com/facebookresearch/optimizers.git@main",
        "pydantic_config" : "pydantic_config @ git+https://github.com/samsja/pydantic_config.git@74c94ee"
    }
                                 
    for match in package_pattern.finditer(uv_lock_content):
        name = match.group("name")
        version = match.group("version")
        dependencies = match.group("dependencies")
        marker = None

        if dependencies:
            marker_match = re.search(r"marker = \"([^\"]+)\"", dependencies)
            if marker_match:
                marker = marker_match.group(1)

            extra_match = re.search(r"extra = \"([^\"]+)\"", dependencies)
            if extra_match:
              extra = extra_match.group(1)
              name = f"{name}[{extra}]"

        if marker:
            if not eval(marker, {"python_full_version": python_version, "sys_platform": platform_system}):
                continue

        is_blacklisted = any([x in name for x in blacklisted])

        if is_blacklisted:
            continue


        is_add_link = any([x in name for x in add_link])

        dependency = f"{name}"
        if is_add_link:
            dependency += f"@{add_link[name]}"
        else:
            dependency += f"=={version}"
        requirements.append(dependency)

    return "\n".join(requirements)

def main():
    with open("uv.lock", "r") as f:
        uv_requirements = f.read()

    python_version = "3.11"  
    platform_system = platform.system().lower()
    if platform_system == "darwin":
        platform_system = "macos"

    requirements_txt = generate_requirements_txt(uv_requirements, python_version, platform_system)

    with open("requirements.txt", "w") as f:
        f.write(requirements_txt)

if __name__ == "__main__":
    main()
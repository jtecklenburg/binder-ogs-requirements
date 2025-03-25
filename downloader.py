import requests
from pathlib import Path
import tarfile

# Subdirectories in Tests/Data on master:
# https://gitlab.opengeosys.org/ogs/ogs/-/tree/master/Tests/Data
subdirs = [
  "TH2M/TH2/heatpipe",
  "Mechanics/EvaluatingBbarWithSimpleExamples"
]

for subdir in subdirs:
    url = f"https://gitlab.opengeosys.org/ogs/ogs/-/archive/master/ogs-master.tar.gz?path=Tests/Data/{subdir}"

    response = requests.get(url, stream=True)
    archive_file = Path(f"{subdir.replace("/", "_")}.tar.gz")

    if archive_file.is_file():
        print(f"{archive_file} already downloaded.")
    else:
        with open(archive_file, mode="wb") as file:
            for chunk in response.iter_content(chunk_size=10 * 1024):
                file.write(chunk)

    strip1 = lambda member, path: member.replace(name=Path(*Path(member.path).parts[3:]))
    if Path(f"Data/{subdir}").is_dir():
        print(f"Path Data/{subdir} already exists.")
    else:
        with tarfile.open(archive_file) as tar:
            tar.extractall(path="Data", filter=strip1)

print("Data downloaded! See 'Data'-folder.")

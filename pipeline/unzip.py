import zipfile
import glob

archive_files = glob.glob("Data/Archive/*.zip")

for archive_file in archive_files:

    with zipfile.ZipFile(archive_file, 'r') as zip_ref:
        zip_ref.extractall('Data/All_Data')

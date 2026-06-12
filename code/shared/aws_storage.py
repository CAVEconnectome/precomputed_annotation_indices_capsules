import logging
import os
import glob
from cloudfiles import CloudFiles

logging.getLogger('urllib3').setLevel(logging.INFO)
logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.INFO)
logging.getLogger('s3transfer').setLevel(logging.INFO)
logging.getLogger('aws-cli').setLevel(logging.INFO)
logging.getLogger('cloudfiles').setLevel(logging.INFO)

def query_folder_on_aws(dir_path, bucket, aws_project_path, display_results=True):
    logging.info("\nquery_folder_on_aws()")

    if dir_path and dir_path[-1] != '/':
        dir_path = dir_path + '/'
    if bucket[-1] != '/':
        bucket = bucket + '/'
    if aws_project_path[-1] != '/':
        aws_project_path = aws_project_path + '/'

    cf = CloudFiles(f"s3://{bucket}{aws_project_path}{dir_path}")
    
    contents = list(cf.list())
    if display_results:
        logging.info(f"\nBucket contents (first 30 shown):\n  {'\n  '.join(contents[:30])}")
    return contents

def upload_folder_to_aws(dir_path, bucket, aws_project_path, dryrun=False):
    logging.info("\nupload_folder_to_aws()")

    if dir_path[-1] != '/':
        dir_path += '/'
    if bucket[-1] != '/':
        bucket += '/'
    if aws_project_path[-1] != '/':
        aws_project_path = aws_project_path + '/'
    
    logging.info(f"upload_folder_to_aws() dir_path: {dir_path}")
    logging.info(f"upload_folder_to_aws() bucket: {bucket}")
    logging.info(f"upload_folder_to_aws() aws_project_path: {aws_project_path}")
    contents = list(glob.glob(f"{dir_path}*"))[:30]
    logging.info(f"upload_folder_to_aws() Top level files/folders to upload (first 30 shown):\n  {'\n  '.join(contents[:30])}")
    contents = list(glob.glob(f"{dir_path}*/*"))[:30]
    logging.info(f"upload_folder_to_aws() 2nd level files/folders to upload (first 30 shown):\n  {'\n  '.join(contents[:30])}")
    contents = list(glob.glob(f"{dir_path}*/*/*"))[:30]
    logging.info(f"upload_folder_to_aws() 3rd level files/folders to upload (first 30 shown):\n  {'\n  '.join(contents[:30])}")
    contents = list(glob.glob(f"{dir_path}*/*/*/*"))[:30]
    logging.info(f"upload_folder_to_aws() 4th level files/folders to upload (first 30 shown):\n  {'\n  '.join(contents[:30])}")
    
    cf = CloudFiles(f"s3://{bucket}{aws_project_path}")

    cf_local = CloudFiles(dir_path)
    if not dryrun:
        cf_local.transfer_to(cf, block_size=64)  # Upload to S3
        # query_folder_on_aws("", bucket, aws_project_path)
    else:
        logging.info("Dry run won't actually upload any files to Amazon Storage")

def download_folder_from_aws(dir_path, dst_path, filename_filter, bucket, aws_project_path, dryrun=False):
    logging.info("\ndownload_folder_from_aws()")

    if dir_path[-1] != '/':
        dir_path += '/'
    if bucket[-1] != '/':
        bucket += '/'
    if aws_project_path[-1] != '/':
        aws_project_path += '/'
    
    if not dst_path:
        dst_path = dir_path
    else:
        if dst_path[-1] != '/':
            dst_path += '/'
    
    logging.info(f"download_folder_from_aws() dir_path: {dir_path}")
    logging.info(f"download_folder_from_aws() dst_path: {dst_path}")
    logging.info(f"download_folder_from_aws() bucket: {bucket}")
    logging.info(f"download_folder_from_aws() aws_project_path: {aws_project_path}")

    contents = query_folder_on_aws(dir_path, bucket, aws_project_path)
    logging.info(f"All bucket contents (first 30 shown):\n  {'\n  '.join(contents[:30])}")
    logging.info(f"B filename_filter: {filename_filter}")
    if filename_filter:
        contents = [c for c in contents if filename_filter in c]
    logging.info(f"Filtered bucket contents:\n  {'\n  '.join(contents)}")
    
    cf = CloudFiles(f"s3://{bucket}{aws_project_path}{dir_path}")

    if not dryrun:
        os.makedirs(f"{dst_path}", exist_ok=True)
        cf_local = CloudFiles(f"{dst_path}")
        cf_local.transfer_from(cf, contents, block_size=64)  # Download from S3

        file_listing = glob.glob(f'{dst_path}/*')
        logging.info(f"\nDownloaded files (first 30 shown) '{dst_path}*' :\n  {'\n  '.join(file_listing[:30])}")
        file_listing = glob.glob(f'{dst_path}/*/*')
        logging.info(f"\nDownloaded files (first 30 shown) '{dst_path}*/*' :\n  {'\n  '.join(file_listing[:30])}")
    else:
        logging.info("Dry run won't actually download any files from Amazon Storage")

def delete_folder_from_aws(dir_path, bucket, aws_project_path):
    logging.info("\ndelete_folder_from_aws()")

    if dir_path[-1] == '/':
        dir_path = dir_path[:-1]
    if bucket[-1] != '/':
        bucket += '/'
    if aws_project_path[-1] != '/':
        aws_project_path += '/'
    
    cf = CloudFiles(f"s3://{bucket}{aws_project_path}")

    if False:
        # Delete the folder directly.
        # This probably won't align with download_folder_from_aws() unless you pass the subdirs in individually since download_folder_from_aws() doesn't create the passed-in folder in the bucket but rather only uploads the contents of the passed-in folder
        logging.info(f"Deleting s3://{bucket}{aws_project_path}/{dir_path}")
        cf.delete(f"s3://{bucket}{aws_project_path}/{dir_path}")
    else:
        # Delete the contents (files or folders) within the specified folder.
        # This appears to have no effect even when the paths are confirmed to be correct.
        # I suspect the problem is associated with a need for manual deletion verification.
        files = []
        for subdir_path in glob.glob(f"../data/{dir_path}/*"):
            subdir_name = os.path.basename(subdir_path)
            for subdir_file_path in glob.glob(f"../data/{dir_path}/{subdir_name}/*"):
                subdir_file_name = os.path.basename(subdir_file_path)
                logging.info(f"Deleting s3://{bucket}{aws_project_path}/{subdir_name}/{subdir_file_name}")
                # cf.delete(f"s3://{bucket}{aws_project_path}/{subdir_name}/{subdir_file_name}")
                files.append(f"s3://{bucket}{aws_project_path}/{subdir_name}/{subdir_file_name}")
            # cf.delete(f"s3://{bucket}{aws_project_path}/{subdir_name}")
            files.append(f"s3://{bucket}{aws_project_path}/{subdir_name}")

        cf.delete(files)
    
    # query_folder_on_aws("", bucket, aws_project_path)

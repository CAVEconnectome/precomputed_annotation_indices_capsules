import logging
import os
import ast
import base64
import time
from google.oauth2 import service_account
from pathlib import Path
from google.cloud.storage import Client, transfer_manager

MAX_WORKERS = 8

def get_storage_client():
    if 'GOOGLE_SECRET' not in os.environ:
        logging.info("'GOOGLE_SECRET' environment variable not found. This can happen when running the pipeline as a local export from Code Ocean.")
        return None
    
    return Client(
        project="em-270621",
        credentials=service_account.Credentials.from_service_account_info(
            ast.literal_eval(
                base64.b64decode(
                    bytes(os.environ['GOOGLE_SECRET'], 'utf-8')).decode('utf-8')
            )
        )
    )

def upload_files_to_gcp(src_dir, filenames, dst_dir, bucket_name, blob_path, dryrun=False):
    """
    Ref: https://docs.cloud.google.com/storage/docs/uploading-objects
    """
    storage_client = get_storage_client()
    if not storage_client:
        return
    if bucket_name[-1] == '/':
        bucket_name = bucket_name[:-1]
    bucket = storage_client.bucket(bucket_name)

    if src_dir[-1] != '/':
        src_dir += '/'
    if blob_path[-1] != '/':
        blob_path += '/'
    if dst_dir[-1] != '/':
        dst_dir += '/'

    total_file_size = 0
    for file_name in filenames:
        file_size = os.path.getsize(src_dir + file_name)
        logging.info(f"Uploading {src_dir + file_name} of size {file_size} B")
        total_file_size += file_size
    logging.info(f"Total file size to be uploaded: {total_file_size} B")

    results = transfer_manager.upload_many_from_filenames(
        bucket, filenames, source_directory=src_dir, blob_name_prefix=blob_path + dst_dir, max_workers=MAX_WORKERS
    )

    logging.info(f"Uploading {len(filenames)} files:\n  {'\n  '.join(sorted(filenames)).strip()}")
    logging.info(f"Destination bucket/blob: {bucket_name}/{blob_path}{dst_dir}")

    if not dryrun:
        for name, result in zip(filenames, results):
            # The results list is either `None` or an exception for each filename in the input list, in order

            if isinstance(result, Exception):
                logging.info("Failed to upload {} due to exception: {}".format(name, result))
            else:
                logging.info("Uploaded {} to {}/{}".format(name, bucket.name, blob_path + dst_dir))
    else:
        logging.info("Dry run won't actually upload any files to Google Storage")

def upload_directory_to_gcp(src_dir, subdir, dst_dir, bucket_name, blob_path, dryrun=False):
    """
    Ref: https://docs.cloud.google.com/storage/docs/uploading-objects
    """
    storage_client = get_storage_client()
    if not storage_client:
        return
    if bucket_name[-1] == '/':
        bucket_name = bucket_name[:-1]
    bucket = storage_client.bucket(bucket_name)

    if src_dir[-1] != '/':
        src_dir += '/'
    if blob_path[-1] != '/':
        blob_path += '/'
    if dst_dir[-1] != '/':
        dst_dir += '/'

    # First, recursively get all files in `directory` as Path objects
    directory_as_path_obj = Path(src_dir + subdir)
    paths = directory_as_path_obj.rglob("*")

    # Filter so the list only includes files, not directories themselves
    file_paths = [path for path in paths if path.is_file()]

    total_file_size = 0
    for file_path in file_paths:
        file_size = os.path.getsize(file_path)
        logging.info(f"Uploading {file_path} of size {file_size} B")
        total_file_size += file_size
    logging.info(f"Total file size to be uploaded: {total_file_size} B")

    # These paths are relative to the current working directory. Next, make them relative to `directory`.
    relative_paths = [path.relative_to(src_dir) for path in file_paths]

    # Finally, convert them all to strings
    string_paths = [str(path) for path in relative_paths]

    # logging.info("paths:", paths)
    # logging.info("file_paths:", file_paths)
    # logging.info("relative_paths:", relative_paths)
    # logging.info("string_paths:", string_paths)
    logging.info(f"Uploading {len(string_paths)} files:\n  {'\n  '.join(sorted(string_paths)).strip()}")
    logging.info(f"Destination bucket/blob: {bucket_name}/{blob_path}{dst_dir}")

    if not dryrun:
        # Start the upload
        results = transfer_manager.upload_many_from_filenames(
            bucket, string_paths, source_directory=src_dir, blob_name_prefix=blob_path + dst_dir, max_workers=MAX_WORKERS
        )

        for name, result in zip(string_paths, results):
            # The results list is either `None` or an exception for each filename in the input list, in order
            if isinstance(result, Exception):
                logging.info("Failed to upload {} due to exception: {}".format(name, result))
            else:
                logging.info("Uploaded {} to {}/{}".format(name, bucket.name, blob_path + dst_dir))
    else:
        logging.info("Dry run won't actually upload any files to Google Storage")

def download_files_from_gcp(sub_dir, dst_dir, filename_filter, bucket_name, blob_path, dryrun=False):
    """
    Ref: https://docs.cloud.google.com/storage/docs/downloading-objects
    """
    storage_client = get_storage_client()
    if not storage_client:
        return
    if bucket_name[-1] == '/':
        bucket_name = bucket_name[:-1]
    bucket = storage_client.bucket(bucket_name)

    if blob_path[-1] != '/':
        blob_path += '/'
    if sub_dir[-1] != '/':
        sub_dir += '/'
    if dst_dir[-1] != '/':
        dst_dir += '/'
    
    blob_names = [blob.name for blob in bucket.list_blobs(prefix=blob_path + sub_dir)]
    logging.info(f"All blob names (first 50 shown):\n  {'\n  '.join(blob_names[:50])}")
    logging.info(f"B filename_filter: {filename_filter}")
    if filename_filter:
        blob_names = [b for b in blob_names if filename_filter in b]
    logging.info(f"Filtered blob names:\n  {'\n  '.join(blob_names)}")
    
    results = transfer_manager.download_many_to_path(
        bucket, blob_names, destination_directory=dst_dir, max_workers=MAX_WORKERS
    )

    # Individual files may fail to pull while others may succeed.
    # Continually try to pull files until we get them all.
    try_list = [blob_name for blob_name in blob_names]
    while True:
        retry_list = []
        for name, result in zip(try_list, results):
            # The results list is either `None` or an exception for each blob in
            # the input list, in order.

            if isinstance(result, Exception):
                print("Failed to download {} due to exception: {}".format(name, result))
                retry_list.append(name)
            else:
                print("Downloaded {} to {}.".format(name, dst_dir + name))
        if not retry_list:
            break
        try_list = retry_list
        logging.info("Some files were not successfully downloaded. Sleeping briefly and trying again.")
        time.sleep(5)
    
    logging.info("All files downloaded from GCP")
